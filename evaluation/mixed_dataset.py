import json
import os
import sqlite3
import sys

import numpy as np
import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from database.interface import get_predictions_by_answer_ids, get_predictions
from evaluation.utils import select_best_roberta_checkpoint, set_label, remove_job_id_from_prompt_mode, get_roc_auc, map_dipper_to_generative_model, highlight_max

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

    # axes
    'axes.spines.right': False,
    'axes.spines.top': False,

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.4,

    # save
    'savefig.format': 'pdf'
})
with open(os.path.join("../datasets", 'mixed_dataset_ids.txt'), 'r', encoding='utf-8') as f:
    ids = [int(line.rstrip('\n')) for line in f]


def gpt_zero_class_predictions():
    gpt_zero_predictions = {}

    for id in ids:
        with open(os.path.join("../results/mixed/gpt-zero", f'gpt-zero-response-{id}.json'), 'r', encoding='utf-8') as f:
            response = json.load(f)
            gpt_zero_predictions[id] = response['documents'][0]['class_probabilities']

    with sqlite3.connect("../../database/database.db") as conn:
        placeholders = ', '.join('?' for _ in ids)
        answers = pd.read_sql(f"""
            SELECT answers.*, jobs.dataset_id, jobs.model, jobs.prompt_mode
            FROM answers
            LEFT JOIN jobs ON answers.job_id = jobs.id
            WHERE answers.id IN ({placeholders})
        """, conn, params=ids, )

    remove_job_id_from_prompt_mode(answers)

    human_answers = answers[answers.is_human == 1]
    improve_answers = answers[answers.prompt_mode == "improve-human"]
    task_answers = answers[answers.prompt_mode == "task"]
    dipper_answers = answers[answers.prompt_mode == "dipper"]

    human = [gpt_zero_predictions[key] for key in human_answers.id.tolist() if key in gpt_zero_predictions]
    df_human = pd.DataFrame(human)
    df_human['source'] = 'human'

    improve = [gpt_zero_predictions[key] for key in improve_answers.id.tolist() if key in gpt_zero_predictions]
    df_improve = pd.DataFrame(improve)
    df_improve['source'] = 'improve'

    task = [gpt_zero_predictions[key] for key in task_answers.id.tolist() if key in gpt_zero_predictions]
    df_task = pd.DataFrame(task)
    df_task['source'] = 'task'

    dipper = [gpt_zero_predictions[key] for key in dipper_answers.id.tolist() if key in gpt_zero_predictions]
    df_dipper = pd.DataFrame(dipper)
    df_dipper['source'] = 'dipper'

    df_gpt = pd.concat([df_human, df_improve, df_task, df_dipper], ignore_index=True)

    df_gpt = df_gpt.melt(id_vars=['source'], var_name='label', value_name='score')

    print(df_gpt.groupby(['source', 'label'])['score'].mean())
    print(df_gpt.groupby(['label'])['score'].mean())

    plt.figure(figsize=(4, 2.5))
    g = sns.barplot(x="source", hue="label", y="score", data=df_gpt, gap=0.1, zorder=2)
    g.legend(title='prediction class', ncol=3, loc="center", bbox_to_anchor=(.5, 1.2), frameon=False)
    plt.ylabel('class prediction')
    plt.xlabel('prompt mode')
    plt.grid(True, zorder=1)
    sns.despine(offset=2, trim=True)
    plt.subplots_adjust(top=0.5)
    plt.tight_layout()
    plt.savefig("plots/gpt-zero-perdictions.pdf")
    # plt.show()


def mixed_dataset_roc_curve():
    df = get_predictions_by_answer_ids("../../database/database.db", ids)
    df = select_best_roberta_checkpoint(df)
    df = df.apply(map_dipper_to_generative_model, axis=1)
    df['generative_model'] = df['generative_model'].fillna('human')
    set_label(df)

    results = []
    for generative_model in ['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct']:
        gen_df = df[df['generative_model'].isin([generative_model, 'human'])]
        for (detector,), sub_df in gen_df.groupby(['detector']):
            fpr, tpr, _ = roc_curve(~sub_df['is_human'], sub_df['prediction'])
            score = auc(fpr, tpr)
            # switch label if score is below 0.5
            if score < 0.5:
                fpr, tpr, _ = roc_curve(df['is_human'], df['prediction'])

            if detector == "fast-detect-gpt":
                detector = "Fast-DetectGPT"
            elif detector == "detect-gpt":
                detector = "DetectGPT"
            elif detector == "intrinsic-dim":
                detector = "Intrinsic-Dim"
            elif detector == "ghostbuster":
                detector = "Ghostbuster"
            elif detector == "roberta":
                detector = "RoBERTa"
            elif detector == "gpt-zero":
                detector = "GPTZero"

            if generative_model == "gpt-4o-mini-2024-07-18":
                generative_model = "GPT-4o-mini"
            elif generative_model == "meta-llama/Llama-3.3-70B-Instruct":
                generative_model = "Llama-3.3-70B-Instruct"

            curve_data = pd.DataFrame({
                "fpr": fpr,
                "tpr": tpr,
                "label": f'{detector}',
                "generative_model": generative_model
            })
            results.append(curve_data)

    df = pd.concat(results, ignore_index=True)

    g = sns.FacetGrid(df, col="generative_model", hue="label", height=2.5, aspect=.9, hue_order=["DetectGPT", "Fast-DetectGPT", "Intrinsic-Dim", "Ghostbuster", "RoBERTa", "GPTZero"])
    g.map(sns.lineplot, "fpr", "tpr", errorbar=None)
    for ax in g.axes.flatten():
        for i in range(5):
            ax.lines[i].set_alpha(0.5)

        label = "GPT-4o-mini" if "GPT" in ax.title.get_text() else "Llama-3.3-70B-Instruct"
        ax.title.set_text(label)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

    handles, labels = g.axes[0, 0].get_legend_handles_labels()
    g.fig.legend(handles, labels, loc='center', ncol=3, bbox_to_anchor=(.5, .93), frameon=False)
    g.fig.tight_layout()
    plt.subplots_adjust(top=0.78)
    for ax in g.axes.flatten():
        ax.grid()
    plt.savefig("plots/gptzero_on_mixed.pdf")

    # plt.show()


def generative_model_comparison():
    df = get_predictions_by_answer_ids("../../database/database.db", ids)
    df = select_best_roberta_checkpoint(df)
    df = df.apply(map_dipper_to_generative_model, axis=1)
    df['generative_model'] = df['generative_model'].fillna('human')
    set_label(df)

    results = []
    for generative_model in ['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct']:
        gen_df = df[df['generative_model'].isin([generative_model, 'human'])]
        for (detector,), sub_df in gen_df.groupby(['detector']):
            results.append({
                "detector": detector,
                "generative_model": generative_model,
                "roc_auc": get_roc_auc(sub_df)
            })

    for (detector,), sub_df in df.groupby(['detector']):
        results.append({
            "detector": detector,
            "generative_model": "Both",
            "roc_auc": get_roc_auc(sub_df)
        })

    df = pd.DataFrame(results)

    pivot_df = df.pivot(columns="generative_model", values="roc_auc", index="detector")
    pivot_df = pivot_df[['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct', 'Both']]
    latex_table = highlight_max(pivot_df).to_latex(
        index=True,
        header=True,
        column_format="lccc",
        escape=False
    )

    print(latex_table)


if __name__ == '__main__':
    mixed_dataset_roc_curve()
    generative_model_comparison()
    gpt_zero_class_predictions()
