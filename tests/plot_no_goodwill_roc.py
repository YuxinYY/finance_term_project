"""Plot all walk-forward test ROC curves for the no-goodwill model run."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'model_artifacts/temporal_no_goodwill'
OUTPUT = ROOT / 'reports/figures/temporal_no_goodwill'
COLORS = {'logistic': ['#a6bddb', '#3680ba', '#034e7b'],
          'lightgbm': ['#fdbb84', '#e66b2c', '#a63603']}

fig, ax = plt.subplots(figsize=(9, 7))
example_curves = {}
for model in ('logistic', 'lightgbm'):
    for index, year in enumerate((2009, 2010, 2011), start=1):
        frame = pd.read_csv(RUN / f'round_{index:02d}' / model / 'test_predictions.csv')
        fpr, tpr, _ = roc_curve(frame['label'], frame['pd_calibrated'])
        score = auc(fpr, tpr)
        if index == 3:
            example_curves[model] = (fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[model][index - 1], linewidth=1.8,
                label=f'{"Logistic regression" if model == "logistic" else "LightGBM"} · {year}')
        print(f'{model} round {index}: AUC={score:.6f}, rows={len(frame):,}')
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Random ranking')
for fpr_target in (0.10, 0.20):
    ax.axvline(fpr_target, color='gray', linestyle=':', linewidth=0.8)
    for model, offset in [('logistic', -13), ('lightgbm', 8)]:
        fpr, tpr = example_curves[model]
        recall = tpr[np.searchsorted(fpr, fpr_target, side='right') - 1]
        color = COLORS[model][2]
        ax.plot(fpr_target, recall, 'o', color=color, markersize=5, zorder=5)
        ax.hlines(recall, 0, fpr_target, colors=color, linestyles='--', linewidth=0.8, alpha=0.75)
        ax.annotate(f'{recall:.1%}', (fpr_target, recall), xytext=(16, offset * 2),
                    textcoords='offset points', fontsize=10, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9),
                    arrowprops=dict(arrowstyle='-', color=color, linewidth=0.7))
ax.text(.02, .97, 'Dashed guides: 2011 test set', transform=ax.transAxes, fontsize=10, va='top')
ax.set(xlabel='False positive rate', ylabel='True positive rate (recall)',
       title='Out-of-time ROC curves: logistic regression vs. LightGBM', xlim=(0, 1), ylim=(0, 1.01))
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.tick_params(axis='both', labelsize=12)
ax.title.set_size(15)
ax.grid(alpha=.2)
ax.legend(loc='lower right', fontsize=11)
fig.tight_layout()
OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / 'roc_comparison_annotated.png', dpi=200)
fig.savefig(OUTPUT / 'roc_comparison_annotated.pdf')
plt.close(fig)
