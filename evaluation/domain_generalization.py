import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score

import seaborn as sns
from tqdm import tqdm

from database.interface import get_predictions
from evaluation.threshold import get_threshold
from evaluation.utils import set_label, map_dipper_to_generative_model, remove_rows_by_condition, \
    remove_job_id_from_prompt_mode

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
    """
    Compares different checkpoints for RoBERTa model and visualize it as two heatmaps.
    """

    all_data = []

    df = get_predictions(
        database="../../database/database.db",
        dataset=None,
        is_human=None,
        detector="roberta",
        # prompt_mode="task",
        max_words=-1
    )

    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource"
    })

    # make sure, to load the correct dipper results
    df = df.apply(map_dipper_to_generative_model, axis=1)

    # set human label of improve and rewrite-human
    set_label(df)

    # rewrite job_id from rewrite-llm and dipper
    remove_job_id_from_prompt_mode(df)

    for group, sub_df in df.groupby(["model_checkpoint", "name"]):
        fpr, tpr, _ = roc_curve(~sub_df['is_human'], sub_df['prediction'])
        roc_auc = auc(fpr, tpr)

        training_dataset = group[0].split("/")[3]
        test_dataset = group[1]

        all_data.append({
            "Training Dataset": training_dataset,
            "Test Dataset": test_dataset,
            "ROC-AUC": roc_auc
        })

    df_all = pd.DataFrame(all_data)
    df_all.replace({"argument-annotated-essays": "AAE"}, inplace=True)
    df_all.replace({"persuade": "PERSUADE"}, inplace=True)

    create_single_plot(df_all, name="domain-generalization-roberta", cmap=sns.light_palette("#E69F00", as_cmap=True))


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

    plt.show()


def zero_shot_threshold_comparison_llm_split():
    for detector in tqdm(["detect-gpt", "fast-detect-gpt", "intrinsic-dim"],
                         desc="Computing zero-shot detection thresholds"):
        all_data = []
        for model in ["gpt-4o-mini-2024-07-18", "meta-llama/Llama-3.3-70B-Instruct"]:
            df = get_predictions(
                database="../../database/database.db",
                dataset=None,
                generative_model=model,
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
                        "Model": model,
                        "Training Dataset": group,
                        "Test Dataset": group2,
                        "F1-Score": score
                    })

        df_all = pd.DataFrame(all_data)
        df_all.replace({"argument-annotated-essays": "AAE"}, inplace=True)
        df_all.replace({"persuade": "PERSUADE"}, inplace=True)
        df_all.replace({"gpt-4o-mini-2024-07-18": "GPT-4o-mini"}, inplace=True)
        df_all.replace({"meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct"}, inplace=True)

        create_heatmap_plot(df_all, f"{detector}_treshold_comparison", "F1-Score")


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
    # zero_shot_threshold_comparison_llm_split()
    zero_shot_threshold_comparison()
