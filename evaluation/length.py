import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from database.interface import get_predictions
from evaluation.utils import (
    select_best_roberta_checkpoint,
    remove_rows_by_condition,
    map_dipper_to_generative_model,
    get_roc_auc,
    get_tpr_at_fpr,
)

# --- Style (wie bei dir, nur was wirklich gebraucht wird) ---
sns.set_theme(context="paper", style=None, font_scale=1, rc={
    "lines.linewidth": 2,
    "legend.handletextpad": .5,
    "legend.handlelength": 1.0,
    "legend.labelspacing": 0.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "ytick.minor.visible": True,
    "ytick.minor.width": 0.3,
    "savefig.format": "pdf",
    "font.family": "Liberation Sans",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20
})

def pastel_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([x + (1 - x) * amount for x in c])

custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
sns.set_palette([pastel_color(c, 0.2) for c in custom_palette])

# --- Helpers ---
def pretty_detector_name(detector: str) -> str:
    mapping = {
        "fast-detect-gpt": "Fast-DetectGPT",
        "detect-gpt": "DetectGPT",
        "intrinsic-dim": "Intrinsic-Dim",
        "ghostbuster": "Ghostbuster",
        "roberta": "RoBERTa",
    }
    return mapping.get(detector, detector)

# --- Load + filter data ---
df = get_predictions(max_words=None, prompt_mode="task")
df = select_best_roberta_checkpoint(df)
df = remove_rows_by_condition(df, conditions={
    "detector": ["gpt-zero", "intrinsic-dim"],
    "name": "mixed"
})
df = df.apply(map_dipper_to_generative_model, axis=1)

max_words_list = [-1, 50, 100, 150, 200, 250]
target_fpr = 0.05

rows_auc = []
rows_tpr = []

for detector, sub_df in df.groupby("detector"):
    for max_words in max_words_list:
        df_subset = sub_df[sub_df["max_words"] == max_words]

        rows_auc.append({
            "detector": pretty_detector_name(detector),
            "number of words": max_words if max_words > 0 else 300,
            "roc_auc": get_roc_auc(df_subset),
        })

        rows_tpr.append({
            "detector": pretty_detector_name(detector),
            "number of words": max_words if max_words > 0 else 300,
            "tpr": get_tpr_at_fpr(df_subset, target_fpr=target_fpr),
        })

df_auc = pd.DataFrame(rows_auc)

print(df_auc)
df_tpr = pd.DataFrame(rows_tpr)

print(df_tpr)

# --- Plot: 2 subplots side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))

marker_style = dict(marker="o", markersize=12, markeredgewidth=0)

# Left: ROC-AUC
g0 = sns.lineplot(
    data=df_auc,
    x="number of words",
    y="roc_auc",
    hue="detector",
    errorbar=None,
    linewidth=5,
    ax=axes[0],
    legend=True,
    **marker_style
)
axes[0].set_xlabel("Maximum Number of Words")
axes[0].set_ylabel("ROC-AUC")
axes[0].grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25, zorder=0)
axes[0].set_xticks([50, 100, 150, 200, 250, 300])

# Right: TPR@5%FPR
g1 = sns.lineplot(
    data=df_tpr,
    x="number of words",
    y="tpr",
    hue="detector",
    errorbar=None,
    linewidth=5,
    ax=axes[1],
    legend=False,
    **marker_style
)
axes[1].set_xlabel("Maximum Number of Words")
axes[1].set_ylabel(f"TPR@5%FPR")
axes[1].grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25, zorder=0)
axes[1].set_xticks([50, 100, 150, 200, 250, 300])

# Single legend
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend_.remove()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.15),
    ncol=min(4, len(labels)),
    frameon=False
)

# limits
# Left plot (ROC-AUC)
axes[0].set_ylim(0.6, 1.0)

# Right plot (TPR)
axes[1].set_ylim(0.0, 1.0)


plt.tight_layout()
plt.subplots_adjust(
    left=0.08,     # weniger Rand links
    right=0.98,    # weniger Rand rechts
    bottom=0.32,   # Platz für Legend
    wspace=0.2    # Abstand zwischen Plots
)
plt.savefig("plots/text_length_auc_and_tpr.pdf")
plt.show()
