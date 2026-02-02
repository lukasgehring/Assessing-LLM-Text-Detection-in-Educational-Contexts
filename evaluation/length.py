import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend

from database.interface import get_predictions
from evaluation.plot_roc_curve import apply_paper_style
from evaluation.utils import (
    select_best_roberta_checkpoint,
    remove_rows_by_condition,
    map_dipper_to_generative_model,
    get_roc_auc,
    get_tpr_at_fpr, set_label,
)


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


def compute_length_comparison():
    # --- Load + filter data ---
    df = get_predictions(max_words=None)

    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "detector": ["gpt-zero", "intrinsic-dim"],
        "name": "mixed"
    })
    df = df.apply(map_dipper_to_generative_model, axis=1)

    set_label(df)

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
    df_tpr = pd.DataFrame(rows_tpr)

    return df_auc, df_tpr


def plot(df_auc, df_tpr):
    # Farben + Linestyles stabil zuordnen
    custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
    palette = [pastel_color(c, 0.2) for c in custom_palette]
    group_order = ["DetectGPT", "Fast-DetectGPT", "Ghostbuster", "RoBERTa"]
    color_map = {g: palette[i] for i, g in enumerate(group_order)}
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    marker_list = ["o", "s", "D", "^", "v", "P", "X"]
    marker_map = {
        g: marker_list[i % len(marker_list)]
        for i, g in enumerate(group_order)
    }
    marker_base = dict(
        markersize=3.5,
        markerfacecolor="white",
        markeredgewidth=1,
    )

    ls_map = {g: linestyles[i % len(linestyles)] for i, g in enumerate(group_order)}
    fig, axes = plt.subplots(1, 2, figsize=(14 / 2.9, 3.5 / 2.9), sharex=True, sharey=False)

    df_auc.sort_values(["number of words"], inplace=True)
    df_tpr.sort_values(["number of words"], inplace=True)

    for g in group_order:
        dsub = df_auc[df_auc["detector"] == g]
        if dsub.empty:
            continue

        axes[0].plot(
            dsub["number of words"].to_numpy(),
            dsub["roc_auc"].to_numpy(),
            color=color_map[g],
            marker=marker_map[g],
            markeredgecolor=color_map[g],
            **marker_base,
            label=str(g),
        )
        axes[0].set_xlabel("Maximum Number of Words")
        axes[0].set_ylabel("ROC-AUC")
        axes[0].grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25, zorder=0)
        axes[0].set_xticks([50, 100, 150, 200, 250, 300])
        axes[0].set_ylim([.55, .95])
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)
        # axes[0].set_yticks([0, 0.5, 1.0])

        dsub = df_tpr[df_tpr["detector"] == g]
        if dsub.empty:
            continue

        axes[1].plot(
            dsub["number of words"].to_numpy(),
            dsub["tpr"].to_numpy(),
            color=color_map[g],
            marker=marker_map[g],
            markeredgecolor=color_map[g],
            **marker_base,
            label=str(g),
        )
        axes[1].set_xlabel("Maximum Number of Words")
        axes[1].set_ylabel(f"TPR@5%FPR")
        axes[1].grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25, zorder=0)
        axes[1].set_xticks([50, 100, 150, 200, 250, 300])
        axes[1].set_ylim([.0, .75])
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["top"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(4, len(labels)),
            frameon=False,
        )
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.85, bottom=.2, top=0.85, wspace=0.25, hspace=0.35)

    plt.savefig("plots/text_length_auc_and_tpr.pdf")
    plt.show()


if __name__ == "__main__":
    apply_paper_style()
    # df_auc, df_tpr = compute_length_comparison()
    # df_auc.to_csv("tmp_1.csv", index=False)
    # df_tpr.to_csv("tmp_2.csv", index=False)

    df_auc = pd.read_csv("tmp_1.csv")
    df_tpr = pd.read_csv("tmp_2.csv")
    plot(df_auc, df_tpr)
