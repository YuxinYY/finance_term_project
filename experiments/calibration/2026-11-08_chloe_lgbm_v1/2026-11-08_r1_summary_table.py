"""Builds the 'what to expect' table (markdown) from metrics_fy2009.csv and bootstrap_vs_b_platt.csv; writes
2026-11-08_summary_table.csv and prints the markdown that is pasted into the round-1 summary."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
m = pd.read_csv(HERE / "2026-11-08_metrics_fy2009.csv")
b = pd.read_csv(HERE / "2026-11-08_bootstrap_vs_b_platt.csv").query("metric == 'd_log_loss'")
b = b.rename(columns={"variant_vs_b_platt": "variant"})[["label", "variant", "point", "ci_low", "ci_high"]]
t = m.merge(b, on=["label", "variant"], how="left")
show = {"raw": "raw", "b_platt": "(b) her Platt (FY2008)", "c_ours": "(c) ours: slope + FY2007 anchor",
        "d_ours_e": "(d) ours + rule e", "iso_val": "isotonic (FY2008)", "iso_val_e": "isotonic + rule e"}
t = t[t.variant.isin(show)].copy()
t.to_csv(HERE / "2026-11-08_summary_table.csv", index=False)
lines = ["| Label | Variant | AUC | Log loss | Δ LL vs (b), 1e-4 [95% CI] | Brier | Mean PD | Observed | Obs/pred | Slope | Intercept |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for lab, labname in (("her_label", "Hers"), ("our_label_s6", "Ours (s6)")):
    for v, name in show.items():
        r = t[(t.label == lab) & (t.variant == v)].iloc[0]
        dll = "0 (ref.)" if v == "b_platt" else f"{r.point*1e4:+.2f} [{r.ci_low*1e4:+.2f}, {r.ci_high*1e4:+.2f}]"
        lines.append(f"| {labname} | {name} | {r.auc:.4f} | {r.log_loss:.5f} | {dll} | {r.brier:.5f} | {r.mean_pd:.3%} | "
                     f"{r.observed_dr:.3%} | {r.ratio_obs_to_pred:.2f} | {r.cal_slope:.3f} | {r.cal_intercept:+.3f} |")
print("\n".join(lines))
