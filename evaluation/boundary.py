import pandas as pd

from database.interface import get_predictions
from evaluation.metrics import get_roc_curve
from evaluation.plot_roc_curve import plot_rocs, apply_paper_style
from evaluation.utils import select_best_roberta_checkpoint, remove_rows_by_condition, set_label, get_roc_auc, get_tpr_at_fpr

MODE_ORDER = ["human", "improve-human", "rewrite-human", "summary", "task+summary"]

MODE_LABEL = {
    "human": "Human",
    "improve-human": "Improve-Human",
    "rewrite-human": "Rewrite-Human",
    "summary": "Summary",
    "task+summary": "Task+Summary",
}

DETECTOR_LABEL = {
    "detect-gpt": "DetectGPT",
    "fast-detect-gpt": "Fast-DetectGPT",
    "ghostbuster": "Ghostbuster",
    "roberta": "RoBERTa",
    "intrinsic-dim": "Intrinsic-Dim",
}


def compute_boundary_comparison(detector=None):
    df = get_predictions(max_words=-1, detector=detector)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={"prompt_mode": "task+resource", "detector": "intrinsic-dim"})

    roc_frames = []
    metric_rows = []

    for detector_name, sub_df in df.groupby("detector", sort=False):

        for mode in MODE_ORDER:
            cur_df = sub_df.copy()

            if mode != "human":
                set_label(cur_df, mode)

            fpr, tpr, _ = get_roc_curve(cur_df, drop_intermediate=True)
            mode_key = mode.replace("\n", "")

            metric_rows.append({
                "detector": detector_name,
                "mode": mode_key,
                "roc_auc": get_roc_auc(cur_df),
                "tpr": get_tpr_at_fpr(cur_df, target_fpr=0.05),
            })

            roc_frames.append(
                pd.DataFrame(
                    {
                        "detector": detector_name,
                        "mode": mode_key,
                        "fpr": fpr,
                        "tpr": tpr,
                    }
                )
            )

    roc_df = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    return roc_df, metrics_df


def _make_latex_table(metrics_df):
    table = (
        metrics_df.pivot_table(
            index="mode",
            columns="detector",
            values=["roc_auc", "tpr"],
        )
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1)
    )

    # "roc_auc" -> "AUC", "tpr" -> "TPR"
    table.columns = pd.MultiIndex.from_tuples(
        [(det, "AUC" if metric == "roc_auc" else "TPR") for det, metric in table.columns]
    )

    table = table.rename(index=MODE_LABEL)
    table = table.rename(columns=DETECTOR_LABEL, level=0)

    return table.to_latex(
        multicolumn=True,
        multicolumn_format="l|",
        column_format="l | cc | cc | cc | cc ",
        float_format=lambda x: f"{x:.3f}".lstrip("0"),
    )


def boundary_comparison():
    roc_df, metrics_df = compute_boundary_comparison()

    print(_make_latex_table(metrics_df))

    roc_df = roc_df.copy()
    roc_df["detector"] = roc_df["detector"].map(DETECTOR_LABEL).fillna(roc_df["detector"])
    roc_df["mode"] = roc_df["mode"].map(MODE_LABEL).fillna(roc_df["mode"])

    apply_paper_style()
    plot_rocs(
        roc_df,
        group_col="detector",
        facet_col="mode",
        group_order=["DetectGPT", "Fast-DetectGPT", "Ghostbuster", "RoBERTa"],
        downsample_step=10,
        xscale="linear",
        xlim=(0, 1),
        ylim=(0, 1.02),
        outfile="plots/boundary_rocs.pdf",
        figsize=(14 / 2.904, 3.5 / 2.904)
    )


if "__main__" == __name__:
    boundary_comparison()
