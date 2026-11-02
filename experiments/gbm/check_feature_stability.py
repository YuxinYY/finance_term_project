"""Did the 15:12 edit of src/features/build.py change the round-1 feature set? Compares the saved GB-01 fold-3 model
(params fitted 2026-10-30) against the CURRENT code: feature list, B.fit output, and fold-3 OOT reproduction."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_lib as L, gbm_lib as G
B = L.B
print("VERSION", B.VERSION, "| n ALL_FEATURES", len(B.ALL_FEATURES), "| has FULL2", hasattr(B, "FULL2_FEATURES"))
art = G.load_model(os.path.join(HERE, "GB-01", "model_fold3"))
print("same feature list as GB-01:", art["features"] == list(B.ALL_FEATURES))
print("extra in current ALL:", sorted(set(B.ALL_FEATURES) - set(art["features"])))
p = L.load_panel(); wf, tr = L.load_splits()
f, trn, te = L.fold_frames(p, wf, tr, 3)
pr = G.predict_pd(art, te)
o = pd.read_parquet(os.path.join(HERE, "GB-01", "oot_predictions.parquet")).query("fold == 3")
print("fold-3 OOT max abs diff, saved params + current transform:", float(np.max(np.abs(te[["row_id"]].assign(p=pr).merge(o, on="row_id").eval("p - pd")))))
fp = B.fit(trn)
old = art["feature_params"]
for k in ("winsor", "categories", "impute_median", "block_share"):
    keys = set(old[k]) & set(fp[k])
    same = all(json.dumps(old[k][c], sort_keys=True) == json.dumps(json.loads(json.dumps(fp[k][c])), sort_keys=True) for c in keys)
    print(k, "n_common", len(keys), "identical on common keys:", same, "| only-new", sorted(set(fp[k]) - set(old[k]))[:12])
