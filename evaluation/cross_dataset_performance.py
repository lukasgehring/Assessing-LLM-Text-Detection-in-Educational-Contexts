import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score

import seaborn as sns
from tqdm import tqdm

from database.interface import get_predictions
from evaluation.threshold import get_threshold
from evaluation.utils import set_label, map_dipper_to_generative_model, remove_rows_by_condition, \
    remove_job_id_from_prompt_mode, get_roc_auc, get_tpr_at_fpr

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


def roberta_checkpoint_comparison():
    all_data = []

    df = get_predictions(
        database="../../database/database.db",
        dataset=None,
        is_human=None,
        #detector="roberta",
        max_words=-1,

    )

    df = remove_rows_by_condition(df, conditions={"prompt_mode": "task+resource", "detector": "intrinsic-dim"})
    df = df.apply(map_dipper_to_generative_model, axis=1)
    set_label(df)
    remove_job_id_from_prompt_mode(df)
    for (detector, name), sub_df in df.groupby(["detector", "name"]):

        if detector == "roberta":
            for (model_checkpoint,), sub_sub_df in sub_df.groupby(["model_checkpoint"]):
                training_dataset = model_checkpoint.split("/")[3]
                test_dataset = name
                all_data.append({
                    "Model": f"Roberta{training_dataset}",
                    "Test Dataset": test_dataset,
                    "auc": get_roc_auc(sub_sub_df),
                    "tpr": get_tpr_at_fpr(sub_sub_df, target_fpr=0.05),
                })
        else:
            test_dataset = name
            all_data.append({
                "Model": detector,
                "Test Dataset": test_dataset,
                "auc": get_roc_auc(sub_df),
                "tpr": get_tpr_at_fpr(sub_df, target_fpr=0.05),
            })







    df_all = pd.DataFrame(all_data)
    df_all.replace({"argument-annotated-essays": "AAE", "persuade": "PERSUADE"}, inplace=True)
    df_all.replace({"Robertaargument-annotated-essays": "RobertaAAE", "Robertapersuade": "RobertaPERSUADE"}, inplace=True)
    df_all.replace({"detect-gpt": "DetectGPT", "fast-detect-gpt": "FastDetectGPT", "ghostbuster":"Ghostbuster"}, inplace=True)
    #df_all["Model"] = "Roberta" + df_all["Training Dataset"]

    row_order = ["RobertaAAE", "RobertaBAWE", "RobertaPERSUADE", "Ghostbuster", "DetectGPT", "FastDetectGPT"]
    col_order = ["AAE", "BAWE", "PERSUADE"]

    # MultiIndex columns: (Testset, Metric)
    pivot = df_all.pivot(index="Model", columns="Test Dataset", values=["auc", "tpr"])
    # Umordnen zu (Testset, Metric) statt (Metric, Testset)
    pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    pivot = pivot.reindex(index=row_order, columns=pd.MultiIndex.from_product([col_order, ["auc", "tpr"]]))
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

    print(r"\setlength{\tabcolsep}{3pt}")
    print(latex)

def create_heatmap_plot(df, name, metric="ROC-AUC"):
    g = sns.FacetGrid(df, col="Model", sharex=True, sharey=True, height=2.5, aspect=1.05)

    vmin = df[metric].min()
    vmax = df[metric].max()

    cbar_ax = g.fig.add_axes([0.86, 0.235, 0.03, 0.625])

    def heatmap(data, **kwargs):
        pivot = data.pivot(index="Training Dataset", columns="Test Dataset", values=metric)
        sns.heatmap(pivot, annot=True, square=True, cmap="Blues", linewidths=1,
                    cbar_kws={'label': metric},
                    vmin=vmin, vmax=vmax,
                    cbar_ax=cbar_ax,
                    **kwargs)

    g.map_dataframe(heatmap)
    g.set_titles(col_template="{col_name}", weight="bold")
    plt.tight_layout(rect=[0.0, 0.0, .95, 1])

    plt.savefig(f"plots/{name}.pdf", format="pdf")

    # plt.show()


def zero_shot_threshold_comparison():
    for detector in tqdm(["detect-gpt", "fast-detect-gpt", "intrinsic-dim"],
                         desc="Computing zero-shot detection thresholds"):
        all_data = []
        df = get_predictions(
            database="../../database/database.db",
            dataset=None,
            is_human=None,
            detector=detector,
            prompt_mode=None,
            max_words=-1
        )

        df = remove_rows_by_condition(df, conditions={
            "prompt_mode": "task+resource"
        })

        # make sure, to load the correct dipper results
        df = df.apply(map_dipper_to_generative_model, axis=1)

        # set human label of improve and rewrite-human
        set_label(df)

        for group, sub_df in df.groupby("name"):
            threshold = get_threshold(sub_df)

            for group2, sub_sub_df in df.groupby("name"):
                score = f1_score(~sub_sub_df['is_human'], sub_sub_df['prediction'] > threshold, average="macro")

                all_data.append({
                    "Training Dataset": group,
                    "Test Dataset": group2,
                    "F1-Score": score
                })

        df_all = pd.DataFrame(all_data)
        df_all.replace({"argument-annotated-essays": "AAE"}, inplace=True)
        df_all.replace({"persuade": "PERSUADE"}, inplace=True)

        create_single_plot(df_all, name=f"domain-generalization-{detector}", metric="F1-Score",
                           cmap=sns.light_palette("#0072B2", as_cmap=True))


def create_single_plot(df, name, metric="ROC-AUC", cmap="Blues"):
    pivot = df.pivot(index="Training Dataset", columns="Test Dataset", values=metric)

    fig, ax = plt.subplots(figsize=(6, 4.7))

    cbar_ax = fig.add_axes([0.78, 0.15, 0.07, 0.8])  # [left, bottom, width, height]

    # Heatmap mit Colorbar in benutzerdefinierter Achse
    sns.heatmap(pivot, annot=True, square=True, cmap=cmap, linewidths=1,
                cbar_kws={'label': metric}, ax=ax, cbar_ax=cbar_ax, vmin=.5, vmax=1)
    # plt.tight_layout(rect=[0.05, 0.0, .8, 1])
    plt.subplots_adjust(right=0.75, bottom=0.1, top=1)
    plt.savefig(f"plots/{name}.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    roberta_checkpoint_comparison()
    # zero_shot_threshold_comparison()
