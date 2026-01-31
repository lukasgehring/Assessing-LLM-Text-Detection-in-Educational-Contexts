import numpy as np
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

    table = (
        df_auc
        .pivot_table(
            index="mode",
            columns="detector",
            values=["roc_auc", "tpr"]
        )
    )

    table = table.swaplevel(0, 1, axis=1).sort_index(axis=1)

    table.columns = pd.MultiIndex.from_tuples(
        [(p, "AUC" if m == "roc_auc" else "TPR") for p, m in table.columns]
    )

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

    df["detector"] = df["detector"].map({
        "fast-detect-gpt": "Fast-DetectGPT",
        "detect-gpt": "DetectGPT",
        "intrinsic-dim": "Intrinsic-Dim",
        "ghostbuster": "Ghostbuster",
        "roberta": "RoBERTa",
    })

    df["mode"] = df["mode"].map({
        "human": "Human",
        "improve-human": "Improve-Human",
        "rewrite-human": "Rewrite-Human",
        "summary": "Summary",
        "task+summary": "Task+Summary",
    })

    apply_paper_style()
    plot_rocs(
        df,
        group_col="detector",
        facet_col="mode",
        downsample_step=10,
        xscale="linear",
        xlim=(0, 1),
        ylim=(0, 1.02),
        outfile="plots/boundary_rocs.pdf",
        figsize=(14 / 2.904, 3.5 / 2.904)
    )


if "__main__" == __name__:
    apply_paper_style()
    boundary_comparison()
