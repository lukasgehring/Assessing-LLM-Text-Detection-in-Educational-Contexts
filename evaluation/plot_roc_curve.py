import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.colors as mcolors


def pastel_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([x + (1 - x) * amount for x in c])


def apply_paper_style():
    sns.set_theme(
        context="paper",
        style="ticks",
        font_scale=1,
        rc={
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 5,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 6,
            "figure.titlesize": 6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "axes.labelpad": 2,
            "axes.titlepad": 2,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.pad": 1,
            "ytick.major.pad": 1,
            "axes.grid": False,
        },
    )


def plot_rocs(
        roc_df,
        *,
        fpr_col="fpr",
        tpr_col="tpr",
        group_col="detector",  # welche Linie?
        facet_col="mode",  # welches Panel?
        group_order=None,
        facet_order=None,
        downsample_step=10,
        xscale="linear",  # "linear" oder "log"
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        diagonal=True,
        grid=True,
        figsize=None,
        outfile=None,
        legend=True,
        legend_ncol=4,
):
    """
    Plottet ROC-Kurven aus einem *long* DataFrame:
    Erwartete Spalten: fpr_col, tpr_col, group_col, facet_col (letztere beiden frei wählbar)

    - 1 Panel pro facet_col-Wert
    - 1 Kurve pro group_col-Wert
    - optional: Downsampling innerhalb jeder (facet, group)-Kurve
    """

    # Orders automatisch bestimmen
    if facet_order is None:
        facet_order = list(roc_df[facet_col].dropna().unique())
    if group_order is None:
        group_order = list(roc_df[group_col].dropna().unique())

    n_panels = len(facet_order)
    if figsize is None:
        figsize = (3.0 * n_panels, 3.2)

    # Farben + Linestyles stabil zuordnen
    custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
    palette = [pastel_color(c, 0.2) for c in custom_palette]
    color_map = {g: palette[i] for i, g in enumerate(group_order)}
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    ls_map = {g: linestyles[i % len(linestyles)] for i, g in enumerate(group_order)}

    # Nur benötigte Spalten (schneller/sauberer)
    df = roc_df[[facet_col, group_col, fpr_col, tpr_col]].copy()

    # Sortierung: wichtig für saubere Linien
    df.sort_values([facet_col, group_col, fpr_col], inplace=True)

    # Downsampling pro Kurve (facet, group)
    if downsample_step and downsample_step > 1:
        df = (
            df.groupby([facet_col, group_col], group_keys=False, sort=False)
            .apply(lambda x: x.iloc[::downsample_step])
        )

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, facet_val in zip(axes, facet_order):
        sub = df[df[facet_col] == facet_val]

        for g in group_order:
            dsub = sub[sub[group_col] == g]
            if dsub.empty:
                continue

            ax.plot(
                dsub[fpr_col].to_numpy(),
                dsub[tpr_col].to_numpy(),
                color=color_map[g],
                linestyle=ls_map[g],
                label=str(g),
            )

        # Chance-Diagonale
        if diagonal:
            if xscale == "log":
                xs = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
                ys = xs
            else:
                xs = np.linspace(xlim[0], xlim[1], 400)
                ys = xs
            ax.plot(xs, ys, linestyle="--", alpha=0.35, color="black", zorder=0)

        ax.set_title(str(facet_val))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xscale(xscale)

        ax.spines["bottom"].set_position(("outward", .5))
        ax.spines["left"].set_position(("outward", .5))

        if grid:
            ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25, zorder=0)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Paper-clean ticks (bei log lieber automatisch)
        if xscale == "linear":
            ax.set_xticks([xlim[0], (xlim[0] + xlim[1]) / 2, xlim[1]])
        ax.set_yticks([0, 0.5, 1.0])

    axes[0].set_ylabel("True Positive Rate")
    for ax in axes:
        ax.set_xlabel("False Positive Rate")

    if legend:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.15),
                ncol=min(legend_ncol, len(labels)),
                frameon=False,
            )

    plt.tight_layout()
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.32 if legend else 0.12, top=0.9, wspace=0.2)

    if outfile:
        plt.savefig(outfile)
    plt.show()
    return fig, axes
