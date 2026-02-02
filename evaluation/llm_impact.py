import pandas as pd

from database.interface import get_predictions
from evaluation.metrics import get_roc_curve
from evaluation.plot_roc_curve import apply_paper_style, plot_rocs
from evaluation.utils import remove_rows_by_condition, select_best_roberta_checkpoint, map_dipper_to_generative_model, set_label, remove_job_id_from_prompt_mode, get_roc_auc, get_tpr_at_fpr

DETECTOR_LABEL = {
    "detect-gpt": "DetectGPT",
    "fast-detect-gpt": "Fast-DetectGPT",
    "ghostbuster": "Ghostbuster",
    "roberta": "RoBERTa",
}

GEN_MODEL_LABEL_PLOT = {
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70b",
    "Both": "Both"
}


def _make_latex_table(metrics_df):
    # Pivot: Zeilen = Detector, Spalten = (Modell, Metrik)
    pivot = metrics_df.pivot(index="detector", columns="generative_model", values=["auc", "tpr"])

    model_order = ['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct', 'Both']

    pivot = pivot.loc[:, (["auc", "tpr"], model_order)]

    # Runden auf 2 Nachkommastellen wie in deinem Beispiel
    pivot = pivot.round(3)

    # Jetzt manuell LaTeX bauen (für exakten Header)
    lines = []
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l | c|c | c|c | c|c}")
    lines.append(r"\toprule")
    lines.append(r"Detector & \multicolumn{2}{l|}{GPT-4o} & \multicolumn{2}{l|}{Llama-3.3} & \multicolumn{2}{l}{Both} \\")
    lines.append(r" & AUC & TPR & AUC & TPR & AUC & TPR\\")
    lines.append(r"\midrule")

    # Zeilen
    for detector in pivot.index:
        row = [detector]
        for model in model_order:
            auc = pivot.loc[detector, ("auc", model)]
            tpr = pivot.loc[detector, ("tpr", model)]

            # Format wie ".82" statt "0.82"
            def fmt(x):
                return f"{x:.3f}".lstrip("0")

            row.extend([fmt(auc), fmt(tpr)])

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    latex_table = "\n".join(lines)
    return latex_table


def compute_model_comparison():
    df = get_predictions(max_words=-1)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={"prompt_mode": "task+resource", "detector": "intrinsic-dim"})

    df = df.apply(map_dipper_to_generative_model, axis=1)
    set_label(df)
    remove_job_id_from_prompt_mode(df)

    roc_frames = []
    metric_rows = []
    for generative_model in ['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct']:
        gen_df = df[df['generative_model'].isin([generative_model, 'human'])]
        for detector_name, sub_df in gen_df.groupby('detector'):
            metric_rows.append({
                "detector": detector_name,
                "generative_model": generative_model,
                "auc": get_roc_auc(sub_df),
                "tpr": get_tpr_at_fpr(sub_df, target_fpr=0.05),
            })

            fpr, tpr, _ = get_roc_curve(sub_df, drop_intermediate=True)
            roc_frames.append(
                pd.DataFrame(
                    {
                        "detector": detector_name,
                        "generative_model": generative_model,
                        "fpr": fpr,
                        "tpr": tpr,
                    }
                )
            )

    for detector_name, sub_df in df.groupby('detector'):
        metric_rows.append({
            "detector": detector_name,
            "generative_model": "Both",
            "auc": get_roc_auc(sub_df),
            "tpr": get_tpr_at_fpr(sub_df, target_fpr=0.05),
        })

        fpr, tpr, _ = get_roc_curve(sub_df, drop_intermediate=True)
        roc_frames.append(
            pd.DataFrame(
                {
                    "detector": detector_name,
                    "generative_model": "Both",
                    "fpr": fpr,
                    "tpr": tpr,
                }
            )
        )

    roc_df = pd.concat(roc_frames, ignore_index=True) if roc_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    return roc_df, metrics_df


def model_comparison():
    roc_df, metrics_df = compute_model_comparison()

    print(_make_latex_table(metrics_df))

    roc_df = roc_df.copy()
    roc_df["detector"] = roc_df["detector"].map(DETECTOR_LABEL).fillna(roc_df["detector"])
    roc_df["generative_model"] = roc_df["generative_model"].map(GEN_MODEL_LABEL_PLOT).fillna(roc_df["generative_model"])

    apply_paper_style()
    plot_rocs(
        roc_df,
        group_col="detector",
        facet_col="generative_model",
        group_order=["DetectGPT", "Fast-DetectGPT", "Ghostbuster", "RoBERTa"],
        facet_order=["GPT-4o-mini", "Llama-3.3-70b", "Both"],
        downsample_step=10,
        xscale="linear",
        xlim=(0, 1),
        ylim=(0, 1.02),
        outfile="plots/generative_model_rocs.pdf",
        figsize=(14 / 2.904, 3.5 / 2.904),
        # legend_anchor_top=1.0,
        adjust_left=.2,
        adjust_right=.8,
        # adjust_top=0.9,
        # adjust_bottom=0.1,
    )


if __name__ == '__main__':
    model_comparison()
