"""Stress tests for the turquoise harness, run via the exact CLI the TAs use."""
import subprocess, sys, os, numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__))
PY = f"{S}/venv/bin/python"; H = f"{S}/unz/harness.py"
base = pd.read_csv(f"{S}/inputs/sample5k.csv", low_memory=False)
ref = pd.read_csv(f"{S}/out/sample5k.csv", header=None)[0].to_numpy()

def run(name, df=None, raw_text=None):
    p_in, p_out = f"{S}/inputs/{name}.csv", f"{S}/out/{name}.csv"
    if raw_text is not None:
        open(p_in, "w").write(raw_text)
    else:
        df.to_csv(p_in, index=False)
    if os.path.exists(p_out): os.remove(p_out)
    r = subprocess.run([PY, H, "--input_csv", p_in, "--output_csv", p_out], cwd="/", capture_output=True, text=True)
    n_in = 0 if df is None else len(df)
    if r.returncode != 0:
        print(f"FAIL  {name}: exit {r.returncode}: {r.stderr.strip().splitlines()[-1]}"); return None
    out = pd.read_csv(p_out, header=None)[0].to_numpy() if os.path.getsize(p_out) else np.array([])
    ok = len(out) == n_in and np.isfinite(out).all() and ((out > 0) & (out < 1)).all() if n_in else len(out) == 0
    print(f"{'PASS' if ok else 'FAIL'}  {name}: rows in={n_in} out={len(out)} range=[{out.min() if len(out) else '-'}, {out.max() if len(out) else '-'}]")
    return out

# 1. Row-order / independence: shuffled input must give the same PD per row
perm = np.random.default_rng(0).permutation(len(base))
o = run("shuffled", base.iloc[perm])
if o is not None: print("      same PD per row after shuffle:", np.allclose(o, ref[perm]))
# 2. Reversed column order
run("cols_reversed", base[base.columns[::-1]])
# 3. Single row, each scored alone must equal batch score (no use of other rows)
singles = []
for i in range(30):
    o = run(f"single_{i}", base.iloc[[i]]) if i < 1 else None
# row-by-row equivalence on 200 rows in one go: compare batch vs. one-firm-year-at-a-time via predict_frame
sys.path.insert(0, f"{S}/unz")
import inference
params, booster = inference.load_parameters()
sub = base.head(300)
batch = inference.predict_frame(sub, params, booster)
alone = np.array([inference.predict_frame(sub.iloc[[i]], params, booster)[0] for i in range(len(sub))])
diff = np.abs(batch - alone) > 1e-12
print(f"{'PASS' if not diff.any() else 'FAIL'}  row-independence: {diff.sum()} of {len(sub)} rows change PD when other rows are removed")
# 4. Messy values
m = base.head(500).copy().astype(object)
m.iloc[0:50, m.columns.get_loc("asst_tot")] = ""
m.iloc[50:100, m.columns.get_loc("eqty_tot")] = "NA"
m.iloc[100:120, m.columns.get_loc("rev_operating")] = "n/a"
m.iloc[120:140, m.columns.get_loc("profit")] = "-"
m.iloc[140:160, m.columns.get_loc("ebitda")] = "inf"
m.iloc[160:180, m.columns.get_loc("HQ_city")] = "XYZ"
m.iloc[180:200, m.columns.get_loc("ateco_sector")] = 999
m.iloc[200:220, m.columns.get_loc("legal_struct")] = "WEIRD"
m.iloc[220:240, m.columns.get_loc("asst_tot")] = 0
m.iloc[240:260, m.columns.get_loc("asst_tot")] = -5
m.iloc[260:280, m.columns.get_loc("id")] = np.nan
m.iloc[280:300, m.columns.get_loc("fs_year")] = np.nan
m.iloc[300:320, :] = np.nan  # fully blank rows
m.iloc[320:340, m.columns.get_loc("roe")] = -9999
m.iloc[340:360, m.columns.get_loc("asst_tot")] = 1e30
run("messy_values", m)
# 5. All-NaN numeric columns
a = base.head(100).copy(); a[[c for c in a.columns if c not in ("id",)]] = np.nan; run("all_nan", a)
# 6. Missing columns the TAs might drop
for col in ["def_date", "stmt_date", "Unnamed: 0", "roa", "roe", "days_rec", "eqty_corp_family_tot", "id", "fs_year", "cf_operations", "HQ_city"]:
    run(f"drop_{col}", base.head(200).drop(columns=[col]))
# 7. Extra columns, label columns
e = base.head(200).copy(); e["default"] = 1; e["junk"] = "x"; run("extra_cols", e)
# 8. Header only
run("header_only", base.head(0))
# 9. Index column written by pandas (unnamed leading column)
p = f"{S}/inputs/with_index.csv"; base.head(200).to_csv(p); r = subprocess.run([PY, H, "--input_csv", p, "--output_csv", f"{S}/out/with_index.csv"], capture_output=True, text=True)
print(("PASS" if r.returncode == 0 else "FAIL") + f"  with_index_column: exit {r.returncode} {r.stderr.strip().splitlines()[-1]}")
# 10. Unseen firm ids & future year
u = base.head(200).copy(); u["id"] = u["id"] + 10**9; u["fs_year"] = 2015; run("unseen_ids_future_year", u)
# 11. Duplicate rows
run("duplicate_rows", pd.concat([base.head(100)] * 3))
# 12. Semicolon / quoted strings in numbers e.g. "1,234"
q = base.head(50).copy().astype(object); q["asst_tot"] = q["asst_tot"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else v); run("thousand_sep", q)
