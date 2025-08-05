import os
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.colors as mcolors
from database.interface import get_predictions
from evaluation.utils import remove_rows_by_condition, select_best_roberta_checkpoint, map_dipper_to_generative_model, \
    set_label

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # lines
    "lines.linewidth": 2,

    # grid
    # "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    'legend.labelspacing': 0.5,

    # axes
    'axes.spines.right': False,
    'axes.spines.top': False,

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.3,

    # save
    'savefig.format': 'pdf',

    # font
    "font.family": "Liberation Sans",
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
})


def pastel_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([x + (1 - x) * amount for x in c])


custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
pastel_colors = [pastel_color(c, 0.2) for c in custom_palette]
sns.set_palette(pastel_colors)


def _f1_score_optimization(df, **kwargs):
    print("F1-score optimization")
    _, _, thresholds = precision_recall_curve(~df.is_human.to_numpy(), df.prediction.to_numpy(), drop_intermediate=True)

    print(len(thresholds))

    best_f1 = 0
    optimal_threshold = 0.5

    for t in thresholds:
        y_pred = (df.prediction.to_numpy() >= t).astype(int)
        macro_f1 = f1_score(~df.is_human.to_numpy(), y_pred, average='macro')
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            optimal_threshold = t
    return optimal_threshold


def _minimal_fp_optimization(df, max_fpr=0.05, **kwargs):
    fpr, tpr, thresholds = roc_curve(~df.is_human, df.prediction)
    optimal_idx = np.where(fpr < max_fpr)[0][-1]
    return thresholds[optimal_idx]


def _j_index_optimization(df, **kwargs):
    fpr, tpr, thresholds = roc_curve(~df.is_human, df.prediction)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def get_threshold(df, invert=False, method="f1-score", **kwargs):
    if method == "f1-score":
        return _f1_score_optimization(df, **kwargs)
    elif method == "minimal-fp":
        return _minimal_fp_optimization(df, **kwargs)
    elif method == "j-index":
        return _j_index_optimization(df, **kwargs)


def plot_threshold_confidence():
    if os.path.isfile("tmp2.csv"):
        df = pd.read_csv("tmp2.csv", index_col=0)
    else:
        df = get_predictions(max_words=-1)
        df = remove_rows_by_condition(df, conditions={
            "detector": ["gpt-zero", "intrinsic-dim"],
            "name": "mixed"
        })

        df = select_best_roberta_checkpoint(df)

        # df = df.apply(map_dipper_to_generative_model, axis=1)
        set_label(df)

        scores = []
        for i, ((detector,), sub_df) in tqdm(enumerate(df.groupby(["detector"])), total=15):

            if detector == "fast-detect-gpt":
                detector = "Fast-DetectGPT"
            elif detector == "detect-gpt":
                detector = "DetectGPT"
            elif detector == "intrinsic-dim":
                detector = "IntrinsicDim"
            elif detector == "ghostbuster":
                detector = "Ghostbuster"
            elif detector == "roberta":
                detector = "RoBERTa"

            for confidence in np.linspace(1e-10, .5, 200):
                threshold = get_threshold(sub_df, method="minimal-fp", max_fpr=confidence)

                acc = f1_score(~sub_df.is_human.to_numpy(), sub_df.prediction >= threshold, average="macro")

                scores.append({
                    "detector": detector,
                    "false positive rate": confidence,
                    "score": acc,
                })

        df = pd.DataFrame(scores)
        df.to_csv("tmp2.csv", index=True)

    plt.figure(figsize=(9, 5.5))

    g = sns.lineplot(
        data=df,
        x="false positive rate",
        y="score",
        hue="detector",
        linewidth=5,
        legend=True,
        zorder=2,
    )
    # g.legend(ncol=4, bbox_to_anchor=(0.199, 1), frameon=False)
    g.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.19),  # center top, above axes
        ncol=4,  # spread horizontally
        frameon=False,  # removes legend border
        columnspacing=1,
    )
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25, zorder=0)
    plt.tick_params(axis='x', which='minor', bottom=False)
    # plt.grid(True, which='minor', linestyle='-', linewidth=0.2, alpha=0.15, zorder=0)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.85, left=0.13)

    plt.ylabel("F1 score", fontweight='bold')
    plt.xlabel("False Positive Rate", fontweight='bold')
    # plt.ylim(0, 1)
    plt.savefig(f'plots/threshold_fpr.pdf')

    plt.show()


def compute_metrics(threshold, data, scores, detector, threshold_type):
    tn, fp, fn, tp = confusion_matrix(~data.is_human.to_numpy(), data.prediction >= threshold).ravel()
    specificity = tn / (tn + fp)
    scores.append({
        "detector": detector,
        "threshold_type": threshold_type,
        "threshold": threshold,
        "acc": (tp + tn) / (tp + tn + fp + fn),
        "specificity": specificity,
        "f1": f1_score(~data.is_human.to_numpy(), data.prediction >= threshold, average="macro"),
    })


def threshold_comparison():
    if os.path.isfile("tmp.csv"):
        df = pd.read_csv("tmp.csv", index_col=0)
    else:

        df = get_predictions(max_words=-1)
        df = remove_rows_by_condition(df, conditions={
            "detector": ["gpt-zero", "intrinsic-dim"],
            "name": "mixed"
        })

        df = select_best_roberta_checkpoint(df)

        set_label(df)

        scores = []
        for i, ((detector,), sub_df) in tqdm(enumerate(df.groupby(["detector"])), total=15):

            if detector == "fast-detect-gpt":
                detector = "Fast-DetectGPT"
            elif detector == "detect-gpt":
                detector = "DetectGPT"
            elif detector == "intrinsic-dim":
                detector = "IntrinsicDim"
            elif detector == "ghostbuster":
                detector = "Ghostbuster"
            elif detector == "roberta":
                detector = "RoBERTa"

            for threshold_type in ["f1-score", "minimal-fp", "j-index"]:
                compute_metrics(
                    threshold=get_threshold(sub_df, method=threshold_type),
                    data=sub_df,
                    scores=scores,
                    detector=detector,
                    threshold_type=threshold_type
                )

            if detector in ["RoBERTa", "Ghostbuster"]:
                compute_metrics(
                    threshold=.5,
                    data=sub_df,
                    scores=scores,
                    detector=detector,
                    threshold_type="static"
                )

        df = pd.DataFrame(scores)

        df.to_csv("tmp.csv", index=True)

    print(df.to_latex(float_format="%.2f", escape=False, column_format="llcccc", index=False, header=True))

    # add barplot
    df = df[df.threshold_type != "static"]
    df.set_index("detector", inplace=True)
    df.drop(columns="threshold", inplace=True)

    df['threshold_type'] = df['threshold_type'].replace({
        "f1-score": "F1 score",
        "minimal-fp": "FPR-based",
        "j-index": "J-Index",
    }, regex=False)

    df = df.melt(id_vars=['threshold_type'], var_name='metric', value_name='score')

    df['metric'] = df['metric'].replace({
        "acc": "Accuracy",
        "specificity": "Specificity",
        "f1": "F1 score",
    }, regex=False)

    plt.figure(figsize=(9, 5.5))
    g = sns.barplot(
        x="metric",
        y="score",
        hue="threshold_type",
        data=df,
        gap=.2,
        width=0.7,
        linewidth=5,
        fill=True,
        zorder=2,
    )
    # g.bar_label(g.containers[0], fontsize=10, fmt="%.2f")
    # g.bar_label(g.containers[1], fontsize=10, fmt="%.2f")
    # g.bar_label(g.containers[2], fontsize=10, fmt="%.2f")
    g.legend(ncol=3, bbox_to_anchor=(.9, 1.19), frameon=False)
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25, zorder=0)
    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.grid(True, which='minor', linestyle='-', linewidth=0.2, alpha=0.15, zorder=0)
    plt.xlabel("Metric", fontweight='bold')
    plt.ylabel("Score", fontweight='bold')
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.9, right=.93)
    plt.savefig(f'plots/threshold_comparison.pdf')
    plt.show()


if __name__ == "__main__":
    df = get_predictions(
        database="../../database/database.db",
        dataset="BAWE",
        prompt_mode="improve-human",
        generative_model="gpt-4o-mini-2024-07-18",
        detector="fast-detect-gpt",
        is_human=None
    )

    threshold_f1 = get_threshold(df, method="f1-score")
    threshold_fp = get_threshold(df, method="minimal-fp", max_fpr=0.05)

    if True:
        sns.kdeplot(data=df, x="prediction", hue="is_human")
        plt.axvline(x=threshold_f1, color="r", linewidth=2, label=f"f1-score {threshold_f1:.2f}")
        plt.axvline(x=threshold_fp, color="b", linewidth=2, label=f"minimal-fp {threshold_fp:.2f}")
        plt.legend()
        plt.show()

    plot_threshold_confidence()
    threshold_comparison()
