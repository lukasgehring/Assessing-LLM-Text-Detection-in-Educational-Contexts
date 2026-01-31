import numpy as np
import pandas as pd

from database.interface import get_predictions
from evaluation.metrics import get_roc_curve
from evaluation.plot_roc_curve import apply_paper_style, plot_rocs
from evaluation.utils import remove_rows_by_condition, select_best_roberta_checkpoint, set_label, filter_and_get_roc_auc, filter_and_get_tpr_at_fpr, get_roc_auc, get_tpr_at_fpr
from evaluation.utils import remove_job_id_from_prompt_mode

PROMPT_ORDER = ["improve-human", "rewrite-human", "summary", "task+summary", "task", "rewrite-llm", "dipper"]

PROMPT_LABEL = {
    "human": "Human",
    "improve-human": r"\makecell[b]{Improve\\Human}",
    "rewrite-human": r"\shortstack{Rewrite\\Human}",
    "summary": "Summary",
    "task+summary": r"\shortstack{Task\\Summary}",
    "task": "Task",
    "rewrite-llm": r"\shortstack{Rewrite\\LLM}",
    "dipper": "Humanize",
}

PROMPT_LABEL_PLOT = {
    "human": "Human",
    "improve-human": "Improve-Human",
    "rewrite-human": "Rewrite-Human",
    "summary": "Summary",
    "task+summary": "Task+Summary",
    "task": "Task",
    "rewrite-llm": "Rewrite-LLM",
    "dipper": "Humanize",
}

DETECTOR_LABEL = {
    "detect-gpt": "DetectGPT",
    "fast-detect-gpt": "Fast-DetectGPT",
    "ghostbuster": "Ghostbuster",
    "roberta": "RoBERTa",
}


def compute_contribution_level_comparison(target_fpr=0.05):
    # get all predictions and choose best robate model
    df = get_predictions(max_words=-1)
    df = select_best_roberta_checkpoint(df)

    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource",
        "detector": ["gpt-zero", "intrinsic-dim"]
    })

    # set human label of improve and rewrite-human
    set_label(df)

    # rewrite job_id from rewrite-llm and dipper
    remove_job_id_from_prompt_mode(df)

    roc_frames = []
    metric_rows = []
    for detector_name, sub_df in df.groupby('detector', sort=False):
        for prompt_mode in sub_df.prompt_mode.unique():

            if prompt_mode == "human":
                continue

            if sub_df[sub_df['prompt_mode'] == prompt_mode].is_human.all():
                filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "task")]
            else:
                filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "human")]

            metric_rows.append(
                {
                    "detector": detector_name,
                    "prompt_mode": prompt_mode,
                    "roc_auc": get_roc_auc(filtered_df),
                    "tpr": get_tpr_at_fpr(filtered_df, target_fpr=target_fpr),
                }
            )

            fpr, tpr, _ = get_roc_curve(filtered_df, drop_intermediate=True)

            roc_frames.append(
                pd.DataFrame(
                    {
                        "detector": detector_name,
                        "prompt_mode": prompt_mode,
                        "fpr": fpr,
                        "tpr": tpr,
                    }
                )
            )

    roc_df = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    return roc_df, metrics_df


def _make_latex_table(metrics_df):
    table = metrics_df.pivot_table(
        index="detector",
        columns="prompt_mode",
        values=["roc_auc", "tpr"]
    ).swaplevel(0, 1, axis=1).sort_index(axis=1)

    table = table[PROMPT_ORDER]

    # (prompt_mode, metric) -> (prompt_mode, AUC/TPR)
    table.columns = pd.MultiIndex.from_tuples(
        [(p, "AUC" if m == "roc_auc" else "TPR") for p, m in table.columns]
    )

    # copy task and set to human
    human_cols = table["task"].copy()
    human_cols.columns = pd.MultiIndex.from_product([["human"], human_cols.columns])
    table = pd.concat([human_cols, table], axis=1)

    table = table.rename(columns=PROMPT_LABEL, level=0)
    table = table.rename(index=DETECTOR_LABEL)

    return table.to_latex(
        multicolumn=True,
        multicolumn_format="l|",
        column_format="l | cc | cc | cc | cc | cc | cc | cc | cc",
        float_format=lambda x: f"{x:.3f}".lstrip("0"),
    )


def detector_overview(target_fpr=0.05):
    """
    Overview of the ROC-AUC of all detectors for each prompt mode over all datasets.

    Output: Latex Table
    """
    roc_df, metrics_df = compute_contribution_level_comparison(target_fpr=target_fpr)

    if metrics_df.empty:
        print("No results to display.")
        return

    print(_make_latex_table(metrics_df))

    roc_df = roc_df.copy()
    roc_df["detector"] = roc_df["detector"].map(DETECTOR_LABEL).fillna(roc_df["detector"])
    roc_df["prompt_mode"] = roc_df["prompt_mode"].map(PROMPT_LABEL_PLOT).fillna(roc_df["prompt_mode"])

    apply_paper_style()
    plot_rocs(
        roc_df,
        group_col="detector",
        facet_col="prompt_mode",
        group_order=["DetectGPT", "Fast-DetectGPT", "Ghostbuster", "RoBERTa"],
        facet_order=["Improve-Human", "Rewrite-Human", "Summary", "Task+Summary", "Task", "Rewrite-LLM", "Humanize"],
        downsample_step=10,
        xscale="linear",
        xlim=(0, 1),
        ylim=(0, 1.02),
        outfile="plots/contribution_levels_rocs.pdf",
        figsize=(13 / 2.904, 7.5 / 2.904),
        nrows=2,
        legend_anchor_bottom=0.1
    )


if __name__ == '__main__':
    detector_overview()
