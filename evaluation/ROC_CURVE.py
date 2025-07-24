import re
import sys

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

from database.interface import get_predictions
from evaluation.utils import get_data, select_best_roberta_checkpoint, remove_rows_by_condition

sns.set_theme(style="darkgrid")
my_palette = sns.color_palette(["#0b84a5", "#f6c85f", "#6f4e7c", "#9dd866", "#ca472f", "#ca472f", "#8dddd0"])


def _roc_plot(df, ax):
    # plot baseline (random classifier)
    points = 100
    ax.plot(np.linspace(0, 1, points), np.linspace(0, 1, points), color='grey', linestyle='--')
    ax.text(.53, .47, "Random Classifier", ha="center", va="center", rotation=45, color="gray", fontsize=8)

    # iterate through prompt modes (except the human one)
    data = []
    for prompt_mode in [pm for pm in df['prompt_mode'].unique() if pm != "human" and pm != ""]:

        # set improve-human to is_human and load task as llm text
        if prompt_mode == "improve-human":
            filtered_df = df[(df['prompt_mode'] == prompt_mode) | (df['prompt_mode'] == "task")]
            filtered_df.loc[df['prompt_mode'] == prompt_mode, 'is_human'] = True
        else:
            filtered_df = df[(df['prompt_mode'] == prompt_mode) | (df['is_human'] == True)]

        # compute roc curve and auc
        fpr, tpr, _ = roc_curve(~filtered_df['is_human'], filtered_df['prediction'], drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        # preprocess prompt mode name
        prompt_mode = prompt_mode.title()
        if bool(re.search(rf"Rewrite-\d+", prompt_mode)):
            prompt_mode = "Rewrite-LLM"
        elif bool(re.search(rf"Dipper-\d+", prompt_mode)):
            prompt_mode = "Dipper-LLM"

        curve_data = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "label": f'{prompt_mode} (AUC = {roc_auc:.2f})'
        })
        data.append(curve_data)
        # plot curve
        # ax.plot(fpr, tpr, lw=2, label=f'{prompt_mode} (AUC = {roc_auc:.2f})')

    df = pd.concat(data, ignore_index=True)

    hue_order = sorted(df.label.unique())

    hue_order = [hue_order[i] for i in [1, 2, 4, 6, 5, 3, 0]]

    sns.lineplot(data=df, x="fpr", y="tpr", hue="label", linewidth=2, hue_order=hue_order)


def roc_plot_single(
        detector: str = 'detect-gpt',
        generative_model: str = "meta-llama/Llama-3.3-70B-Instruct",
        dataset: str = "argument-annotated-essays",
        log_scale: bool = False,
):
    """
    Create a ROC plot for a single detector, dataset and generative model.

    :param detector: Detector name, e.g. 'detect-gpt'
    :param generative_model: Generative model name, e.g. 'meta-llama/Llama-3.3-70B-Instruct'
    :param dataset: Dataset name, e.g. 'argument-annotated-essays'
    :param log_scale: Log scale of ROC curve
    """
    # create figure
    plt.figure(figsize=(4.5, 5))

    # load data
    df = get_data(generative_model=generative_model, dataset=dataset, detector=detector)
    df = remove_rows_by_condition(df, {
        'prompt_mode': "task+resource",
    })
    df = select_best_roberta_checkpoint(df[df['name'] == dataset], dataset=dataset)
    ax = plt.gca()

    # create plot
    _roc_plot(df, ax)

    # setup figure
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve - {detector.title()}')
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8,
        labelspacing=.5,
        handlelength=.75,
        borderpad=0.75
    )

    if log_scale:
        plt.xscale('log')
    else:
        plt.axis("square")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def roc_plot_multi(roberta_checkpoints: dict = None, generative_model: str = "meta-llama/Llama-3.3-70B-Instruct",
                   save_plot: bool = False):
    """
    Create a ROC plot for all detectors, and datasets for a generative model.

    :param roberta_checkpoints: dictionary with keys 'dataset', and value 'model_checkpoint'
    :param generative_model: Name of generative model, e.g. 'meta-llama/Llama-3.3-70B-Instruct'
    :param save_plot: If True, save the figure
    """
    # load data
    df = get_data(generative_model)

    nrows, ncols = 3, 4
    # create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.25 * ncols, 3.4 * nrows), sharex=True, sharey=True)

    # iterate trough all dataset, detector combinations
    for i, ((dataset, detector), sub_df) in enumerate(df.groupby(["name", "detector"])):
        if detector == "roberta":
            if roberta_checkpoints is None:
                sub_df = select_best_roberta_checkpoint(sub_df[sub_df['name'] == dataset], dataset)
            else:
                sub_df = sub_df[sub_df.model_checkpoint == roberta_checkpoints[dataset]]

        row_idx = i // 4
        col_idx = i % 4
        ax = axes[row_idx, col_idx]

        # create plot
        _roc_plot(sub_df, ax)

        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_aspect('equal', 'box')

        # Debug
        # ax.legend()

        if row_idx == 0:
            ax.set_title(detector.title(), fontsize=12, weight='bold')

        if col_idx == 0:
            ax.set_ylabel(dataset.title(), fontsize=12, weight='bold')

    # setup figure
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [label.split("(AUC")[0].strip() for label in labels]

    plt.subplots_adjust(bottom=0.11, top=0.95, left=0.05, right=0.98)

    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        bbox_to_anchor=(.5, .0),
        ncol=len(labels),
        fontsize=11,
        labelspacing=.5,
        handlelength=.75,
        borderpad=0.75
    )

    if save_plot:
        plt.savefig(f"plots/roc_curve_multi_{generative_model.split('/')[-1]}.pdf", format="pdf",
                    transparent=False)

    plt.show()


if __name__ == "__main__":
    detect_gpt = "detect-gpt"
    intrinsic_dim = "intrinsic-dim"
    roberta = "roberta"
    ghostbuster = "ghostbuster"
    fast_detect_gpt = "fast-detect-gpt"

    llama = "meta-llama/Llama-3.3-70B-Instruct"
    gpt = "gpt-4o-mini-2024-07-18"

    aae = "argument-annotated-essays"
    bawe = "BAWE"
    persuade = "persuade"

    roc_plot_single(detector=roberta, generative_model=gpt, dataset=bawe, log_scale=False)
    sys.exit(0)
    roc_plot_multi(
        # roberta_checkpoints={
        #    'argument-annotated-essays': 'detectors/RoBERTa/checkpoints/persuade/binary/checkpoint-24',
        #    'persuade': 'detectors/RoBERTa/checkpoints/argument-annotated-essays/binary/checkpoint-123',
        #    'BAWE': 'detectors/RoBERTa/checkpoints/argument-annotated-essays/binary/checkpoint-123',
        # },
        generative_model=llama,
        save_plot=True
    )
