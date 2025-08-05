import re

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

from database.interface import get_predictions
from evaluation.utils import get_data, remove_rows_by_condition, select_best_roberta_checkpoint, get_roc_auc, set_label, compute_mean_std, get_pr_auc, map_dipper_to_generative_model, \
    filter_and_get_roc_auc, highlight_max
from evaluation.utils import remove_job_id_from_prompt_mode

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # lines
    "lines.linewidth": 2,

    # grid
    "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    'legend.labelspacing': 0.5,
    "legend.loc": "upper center",

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.4,

    # save
    'savefig.format': 'pdf'
})


def get_dataframe(generative_models):
    results = []
    for generative_model in generative_models:
        df = get_data(generative_model=generative_model)

        df = remove_rows_by_condition(df, conditions={
            "name": "mixed",
            "detector": "gpt-zero"
        })

        for i, ((dataset, detector), sub_df) in enumerate(df.groupby(["name", "detector"])):
            if detector == "roberta":
                sub_df = select_best_roberta_checkpoint(sub_df)

            for prompt_mode in [pm for pm in sub_df['prompt_mode'].unique() if
                                pm != "human" and pm != "" and pm != "task+resource"]:

                # set improve-human to is_human and load task as llm text
                if prompt_mode == "improve-human":
                    filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "task")]
                    filtered_df.loc[sub_df['prompt_mode'] == prompt_mode, 'is_human'] = True
                else:
                    filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['is_human'] == True)]

                # compute roc curve and auc
                fpr, tpr, _ = roc_curve(~filtered_df['is_human'], filtered_df['prediction'])
                roc_auc = auc(fpr, tpr)

                model_short = "GPT-4o-mini" if "gpt-4o" in generative_model else "LLaMA-3.3-70B-Instruct"

                results.append({
                    "Model": model_short,
                    "Dataset": dataset,
                    "Prompt Mode": prompt_mode,
                    "Detector": detector,
                    "ROC-AUC": f"{roc_auc:.3f}",
                    "FPR": fpr,
                })

    df_results = pd.DataFrame(results)

    df_results['Detector'] = df_results['Detector'].replace(
        ["detect-gpt", "intrinsic-dim", "ghostbuster", "roberta", "fast-detect-gpt"],
        ["DetectGPT", "Intrinsic-Dim", "Ghostbuster", "RoBERTa", "Fast-DetectGPT"]
    )

    df_results['Dataset'] = df_results['Dataset'].replace(
        ["persuade", "argument-annotated-essays"],
        ["PERSUADE", "AAE"]
    )

    df_results['Prompt Mode'] = df_results['Prompt Mode'].replace(to_replace=r'^dipper-\d+$', value='dipper',
                                                                  regex=True)
    df_results['Prompt Mode'] = df_results['Prompt Mode'].replace(to_replace=r'^rewrite-\d+$', value='rewrite',
                                                                  regex=True)
    df_results['ROC-AUC'] = pd.to_numeric(df_results['ROC-AUC'], errors='coerce')
    return df_results


def catplot():
    df = get_predictions(max_words=-1)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource"
    })

    # make sure, to load the correct dipper results
    df = df.apply(map_dipper_to_generative_model, axis=1)

    # set human label of improve and rewrite-human
    set_label(df)

    # rewrite job_id from rewrite-llm and dipper
    remove_job_id_from_prompt_mode(df)

    results = []
    for generative_model in ['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct']:
        gen_df = df[df['generative_model'].isin([generative_model, 'human'])]
        for (detector, dataset), sub_df in gen_df.groupby(['detector', 'name']):
            for prompt_mode in sub_df.prompt_mode.unique():
                roc_auc = filter_and_get_roc_auc(sub_df, prompt_mode)
                if roc_auc is not None:
                    results.append({
                        "detector": detector,
                        "prompt_mode": prompt_mode,
                        "dataset": dataset,
                        "model": generative_model,
                        "roc_auc": roc_auc
                    })

    df = pd.DataFrame(results)

    df.sort_values(by=['model', 'dataset', 'detector'], inplace=True)

    df['mixed'] = df['dataset'] + ' - ' + df['detector'] + ' - ' + df['prompt_mode']

    grid = sns.FacetGrid(df, col='model', height=4, aspect=1.4, sharex=False)

    detector_order = df['detector'].unique()

    grid.map_dataframe(sns.scatterplot, 'mixed', 'roc_auc', 'detector', style="prompt_mode", hue="detector",
                       zorder=2, s=80, alpha=0.8, hue_order=detector_order)

    grid.add_legend(bbox_to_anchor=(1, 0.525), loc='center right', borderaxespad=0.)

    for ax in grid.axes.flat:
        xticklabels = ax.get_xticklabels()
        labels = [label.get_text().split(' | ')[0] for label in xticklabels]

        new_labels = [label.split(" - ")[0] for i, label in enumerate(labels) if i % 35 == 0]

        ax.set_xticks([17, 52, 87])
        ax.set_xticklabels(new_labels)

        ax.axvline(x=34.5, color='gray', linestyle='--', zorder=1)
        ax.axvline(x=69.5, color='gray', linestyle='--', zorder=1)

    grid.set_xlabels("Dataset")
    grid.set_titles("{col_name}")

    grid.set(ylim=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 0.87, 1])
    for ax in grid.axes.flatten():
        ax.grid()

    plt.savefig("plots/results_overview.pdf", format="pdf")

    plt.show()


def detector_overview():
    """
    Overview of the ROC-AUC of all detectors for each prompt mode over all datasets.

    Output: Latex Table
    """

    # get all predictions and choose best robate model
    df = get_predictions(max_words=-1)
    df = select_best_roberta_checkpoint(df)

    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource"
    })

    # set human label of improve and rewrite-human
    set_label(df)

    # rewrite job_id from rewrite-llm and dipper
    remove_job_id_from_prompt_mode(df)

    results = []
    for (detector,), sub_df in df.groupby(['detector']):
        for prompt_mode in sub_df.prompt_mode.unique():
            roc_auc = filter_and_get_roc_auc(sub_df, prompt_mode)
            if roc_auc is not None:
                results.append({
                    "detector": detector,
                    "prompt_mode": prompt_mode,
                    "roc_auc": roc_auc
                })
        results.append({
            "detector": detector,
            "prompt_mode": "all",
            "roc_auc": get_roc_auc(sub_df)
        })
    df = pd.DataFrame(results)

    # create pivot dataframe
    df_pivot = df.pivot(index='detector', columns='prompt_mode', values='roc_auc')
    order = ["improve-human", "rewrite-human", "summary", "task+summary", "task", "rewrite-llm", "dipper", "all"]
    df_pivot = df_pivot[order]
    df_pivot.index.name = None

    # df_pivot['mean'] = df_pivot.mean(axis=1)
    with pd.option_context('display.max_columns', None):
        print(df_pivot)
    # create latex table
    latex_table = highlight_max(df_pivot).to_latex(index=True, float_format="%.2f", column_format='lcccccccc', escape=False)
    print(latex_table)


if __name__ == '__main__':
    catplot()
    detector_overview()
