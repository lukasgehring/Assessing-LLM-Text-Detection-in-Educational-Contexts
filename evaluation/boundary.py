import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from database.interface import get_predictions
from evaluation.length import pretty_detector_name
from evaluation.metrics import get_roc_curve
from evaluation.utils import select_best_roberta_checkpoint, remove_rows_by_condition, map_dipper_to_generative_model, set_label, get_roc_auc, get_tpr_at_fpr

import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors

def pastel_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([x + (1 - x) * amount for x in c])

def apply_paper_style():
    # Seaborn Theme (setzt auch mpl.rcParams)
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

    # Palette wie bei dir (pastell)
    custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
    sns.set_palette([pastel_color(c, 0.2) for c in custom_palette])

    # Ein paar mpl Defaults, die für Paper nice sind
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def plot_boundary_rocs_paper(
    df_roc,
    detectors_order=None,
    mode_order=None,
    outfile="plots/boundary_rocs.pdf",
    figsize=(14, 3.5),          # ähnlich wie dein 2-Panel plot (breit und flach)
    downsample_step=10,         # macht es schneller + kleinere PDF, optisch meist identisch
    target_fpr=None,            # optional: vertikale Linie bei z.B. 0.05
):
    """
    df_roc columns: detector, fpr, tpr, mode
    - 5 Panels nebeneinander (mode als Facet)
    - 1 Kurve pro detector
    - Style: passend zu deinem Seaborn-Paper-Setup
    """

    if mode_order is None:
        mode_order = ["human", "improve-human", "rewrite-human", "summary", "task+summary"]

    if detectors_order is None:
        detectors_order = list(df_roc["detector"].unique())

    # sortieren (wichtig für saubere Linien)
    df_roc = df_roc.sort_values(["mode", "detector", "fpr"])

    # optional: downsampling pro Kurve
    if downsample_step and downsample_step > 1:
        df_roc = (
            df_roc.groupby(["mode", "detector"], group_keys=False)
                 .apply(lambda x: x.iloc[::downsample_step])
        )

    # Farben aus aktueller sns-Palette ziehen (damit es 1:1 matcht)
    palette = sns.color_palette(n_colors=len(detectors_order))
    color_map = {det: palette[i] for i, det in enumerate(detectors_order)}

    fig, axes = plt.subplots(1, len(mode_order), figsize=figsize, sharex=True, sharey=True)

    if len(mode_order) == 1:
        axes = [axes]

    # Plot
    for ax, mode in zip(axes, mode_order):
        sub = df_roc[df_roc["mode"] == mode]

        for det in detectors_order:
            dsub = sub[sub["detector"] == det]
            if dsub.empty:
                continue
            ax.plot(
                dsub["fpr"].to_numpy(),
                dsub["tpr"].to_numpy(),
                linewidth=3,                 # wie bei dir in lineplot
                color=color_map[det],
                label=det
            )

        # Chance diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2, alpha=0.35, color="black", zorder=0)

        # optional: vertikale Linie bei target_fpr
        #if target_fpr is not None:
        #    ax.axvline(target_fpr, linestyle=":", linewidth=2, alpha=0.6, color="black", zorder=0)

        ax.set_title(mode)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Grid exakt wie bei dir
        ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25, zorder=0)

        # Spines wie bei dir (rechts/oben aus)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Wenige Ticks -> paper clean
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])

    # Achsenbeschriftungen: wie “aus einem Guss”, sparsam
    axes[0].set_ylabel("TPR")
    for ax in axes:
        ax.set_xlabel("FPR")

    # Single Legend unten (ähnlich wie dein Plot)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=min(4, len(labels)),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.06,
        right=0.98,
        bottom=0.32,   # Platz für Legend wie bei dir
        top=0.9,
        wspace=0.25
    )

    if outfile:
        plt.savefig(outfile)
    plt.show()

    return fig, axes


def compute_boundary_comparison(detector=None):
    df = get_predictions(max_words=-1, detector=detector)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource",
        "detector": "intrinsic-dim",
    })

    desired_order = ["human", "improve-human", "rewrite-human", "summary", "task+summary"]

    data = []
    auc_scores = []
    for detector, sub_df in df.groupby("detector"):
        for mode in desired_order:
            if mode != "human":
                set_label(sub_df, mode)

            sub_df['is_human'] = sub_df['is_human'] | (sub_df['prompt_mode'] == mode)

            fpr, tpr, _ = get_roc_curve(sub_df, drop_intermediate=True)
            model_label = mode.replace('\n', '')

            num_human = sub_df['is_human'].sum()
            num_llm = (~sub_df['is_human']).sum()

            sub_df['weights'] = np.where(sub_df['is_human'], 1, num_human / num_llm)

            auc_scores.append({
                'detector': detector,
                'roc_auc': get_roc_auc(sub_df),
                'tpr': get_tpr_at_fpr(sub_df, target_fpr=0.05),
                'mode': f"{model_label}"
            })

            for x, y in zip(fpr, tpr):
                data.append({
                    'detector': detector,
                    'fpr': x,
                    'tpr': y,
                    'mode': f"{model_label}"
                })

    df = pd.DataFrame(data)
    df_auc = pd.DataFrame(auc_scores)

    return df, df_auc


def boundary_comparison():
    df, df_auc = compute_boundary_comparison()

    df_pivot = df_auc.pivot(values='roc_auc', index=['mode'], columns=['detector'])
    print(df_pivot.to_latex(header=True, index=True, float_format='%.3f'))

    table = (
        df_auc
        .pivot_table(
            index="mode",
            columns="detector",
            values=["roc_auc", "tpr"]
        )
    )

    table = table.swaplevel(0, 1, axis=1).sort_index(axis=1)

    #order = ["improve-human", "rewrite-human", "summary", "task+summary", "task", "rewrite-llm", "dipper"]
    #table = table[order]

    table.columns = pd.MultiIndex.from_tuples(
        [(p, "AUC" if m == "roc_auc" else "TPR") for p, m in table.columns]
    )

    #human_cols = table["task"].copy()

    #human_cols.columns = pd.MultiIndex.from_product([["human"], human_cols.columns])

    #table = pd.concat([human_cols, table], axis=1)

    row_rename_map = {
        "human": "Human",
        "improve-human": "Improve-Human",
        "rewrite-human": "Rewrite-Human",
        "summary": "Summary",
        "task+summary": "Task+Summary",
    }
    table = table.rename(index=row_rename_map)

    col_rename_map = {
        "detect-gpt": "DetectGPT",
        "fast-detect-gpt": "Fast-DetectGPT",
        "ghostbuster": "Ghostbuster",
        "roberta": "RoBERTa",
    }
    table = table.rename(columns=col_rename_map, level=0)



    latex = table.to_latex(
        multicolumn=True,
        multicolumn_format="l|",
        column_format="l | cc | cc | cc | cc ",
        float_format=lambda x: f"{x:.3f}".lstrip("0"),
    )

    print(latex)

    df["detector"] = df["detector"].map(pretty_detector_name)

    plot_boundary_rocs_paper(
        df,
        detectors_order=["DetectGPT", "Fast-DetectGPT", "RoBERTa", "Ghostbuster"],  # optional, falls du Ordnung willst
        outfile="plots/boundary_rocs.pdf",
        figsize=(14, 3.5),
        downsample_step=10,
        target_fpr=0.05,  # optional, wenn du die 5% FPR visually markieren willst
    )
    # oder als PDF:
    #plot_boundary_rocs_compact(df, outfile="roc_boundaries.pdf")


    return

    grid = sns.FacetGrid(df, col="detector", hue="mode", height=2.5, aspect=1, sharex=False, sharey=False, col_wrap=2,
                         legend_out=False)

    grid.map(sns.lineplot, "fpr", "tpr", errorbar=None)

    for i, ax in enumerate(grid.axes.flatten()):
        ax.grid(True)  # Gitterlinien aktivieren
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    title_map = {
        "detect-gpt": "DetectGPT",
        "fast-detect-gpt": "FastDetectGPT",
        "ghostbuster": "Ghostbuster",
        "roberta": "RoBERTa",
        "intrinsic-dim": "IntrinsicDim",
    }
    for ax, title in zip(grid.axes.flatten(), grid.col_names):
        new_title = title_map.get(title, title)
        ax.set_title(new_title, weight='bold')

    grid.add_legend(title="Label Boundary")
    sns.move_legend(grid, "upper left", bbox_to_anchor=(.63, .26), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(f"plots/label_boundaries_roc_all.pdf")
    plt.show()


def boundary_comparison_single(detector):
    df, _ = compute_boundary_comparison(detector)

    plt.figure(figsize=(4, 3.8))

    grid = sns.lineplot(data=df, x="fpr", y="tpr", errorbar=None, hue="mode")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.text(.53, .47, s="Random Classifier", rotation=45, horizontalalignment='center',
             verticalalignment='center', color="gray", weight='bold')

    grid.set_xlabel('False Positive Rate')
    grid.set_ylabel('True Positive Rate')

    sns.despine(offset=0, trim=True)
    plt.gca().set_aspect('equal')

    plt.tight_layout()

    plt.savefig(f"plots/label_boundaries_roc_{detector}.pdf")
    # plt.show()


if "__main__" == __name__:
    boundary_comparison()
    # boundary_comparison_single(detector="detect-gpt")
