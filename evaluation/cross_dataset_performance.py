import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score

import seaborn as sns
from tqdm import tqdm

from database.interface import get_predictions
from evaluation.metrics import get_roc_curve
from evaluation.plot_roc_curve import apply_paper_style, plot_rocs
from evaluation.threshold import get_threshold
from evaluation.utils import set_label, map_dipper_to_generative_model, remove_rows_by_condition, \
    remove_job_id_from_prompt_mode, get_roc_auc, get_tpr_at_fpr

ROW_ORDER = ["RobertaAAE", "RobertaBAWE", "RobertaPERSUADE", "Ghostbuster", "DetectGPT", "FastDetectGPT"]
COL_ORDER = ["AAE", "BAWE", "PERSUADE"]


def _make_latex_table(metrics_df):
    # MultiIndex columns: (Testset, Metric)
    pivot = metrics_df.pivot(index="Model", columns="Test Dataset", values=["auc", "tpr"])
    # Umordnen zu (Testset, Metric) statt (Metric, Testset)
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    pivot = pivot.reindex(index=ROW_ORDER, columns=pd.MultiIndex.from_product([COL_ORDER, ["auc", "tpr"]]))
    pivot = pivot.round(3)

    def fmt(x):
        return f"{x:.3f}".lstrip("0")

    pivot = pivot.applymap(fmt)

    latex = pivot.to_latex(
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        column_format="l" + "c" * pivot.shape[1],
    ).replace("0.", ".")

    return r"\setlength{\tabcolsep}{3pt}" + "\n" + latex


def compute_dataset_comparison():
    df = get_predictions(
        database="../../database/database.db",
        dataset=None,
        is_human=None,
        max_words=-1,

    )

    df = remove_rows_by_condition(df, conditions={"prompt_mode": "task+resource", "detector": "intrinsic-dim"})
    df = df.apply(map_dipper_to_generative_model, axis=1)
    set_label(df)
    remove_job_id_from_prompt_mode(df)

    roc_frames = []
    metric_rows = []

    for (detector_name, dataset), sub_df in df.groupby(["detector", "name"]):

        if detector_name == "roberta":
            for (model_checkpoint,), sub_sub_df in sub_df.groupby(["model_checkpoint"]):
                training_dataset = model_checkpoint.split("/")[3]
                test_dataset = dataset
                metric_rows.append({
                    "Model": f"Roberta{training_dataset}",
                    "Test Dataset": test_dataset,
                    "auc": get_roc_auc(sub_sub_df),
                    "tpr": get_tpr_at_fpr(sub_sub_df, target_fpr=0.05),
                })

                fpr, tpr, _ = get_roc_curve(sub_df, drop_intermediate=True)
                roc_frames.append(
                    pd.DataFrame(
                        {
                            "detector": f"Roberta{training_dataset}",
                            "dataset": dataset,
                            "fpr": fpr,
                            "tpr": tpr,
                        }
                    )
                )
        else:
            test_dataset = dataset
            metric_rows.append({
                "Model": detector_name,
                "Test Dataset": test_dataset,
                "auc": get_roc_auc(sub_df),
                "tpr": get_tpr_at_fpr(sub_df, target_fpr=0.05),
            })

            fpr, tpr, _ = get_roc_curve(sub_df, drop_intermediate=True)
            roc_frames.append(
                pd.DataFrame(
                    {
                        "detector": detector_name,
                        "dataset": test_dataset,
                        "fpr": fpr,
                        "tpr": tpr,
                    }
                )
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.replace({"argument-annotated-essays": "AAE", "persuade": "PERSUADE"}, inplace=True)
    metrics_df.replace({"Robertaargument-annotated-essays": "RobertaAAE", "Robertapersuade": "RobertaPERSUADE"}, inplace=True)
    metrics_df.replace({"detect-gpt": "DetectGPT", "fast-detect-gpt": "FastDetectGPT", "ghostbuster": "Ghostbuster"}, inplace=True)

    roc_df = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()

    roc_df.replace({"argument-annotated-essays": "AAE", "persuade": "PERSUADE"}, inplace=True)
    roc_df.replace({"Robertaargument-annotated-essays": "RobertaAAE", "Robertapersuade": "RobertaPERSUADE"}, inplace=True)
    roc_df.replace({"detect-gpt": "DetectGPT", "fast-detect-gpt": "FastDetectGPT", "ghostbuster": "Ghostbuster"}, inplace=True)

    return roc_df, metrics_df


def dataset_comparison():
    roc_df, metrics_df = compute_dataset_comparison()

    print(_make_latex_table(metrics_df))

    apply_paper_style()
    plot_rocs(
        roc_df,
        group_col="dataset",
        facet_col="detector",
        # group_order=["DetectGPT", "Fast-DetectGPT", "Ghostbuster", "RoBERTa"],
        # facet_order=["GPT-4o-mini", "Llama-3.3-70b", "Both"],
        downsample_step=10,
        xscale="linear",
        xlim=(0, 1),
        ylim=(0, 1.02),
        outfile="plots/dataset_rocs.pdf",
        figsize=(13 / 2.904, 10 / 2.904),
        nrows=2, legend_anchor_bottom=0.1
    )


if __name__ == "__main__":
    dataset_comparison()
