import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

from database.interface import get_predictions
from evaluation.metrics import get_roc_curve
from evaluation.utils import select_best_roberta_checkpoint, remove_rows_by_condition, map_dipper_to_generative_model

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


def compute_boundary_comparison(detector=None):
    df = get_predictions(max_words=-1, detector=detector)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource",
        # "detector": "intrinsic-dim",
    })

    df = df.apply(map_dipper_to_generative_model, axis=1)

    df['generative_model'] = df['generative_model'].fillna("human")

    df['prompt_mode'] = df['prompt_mode'].replace({
        "improve-human": "Improved-\nHuman",
        "rewrite-human": "Rewrite-\nHuman",
        "human": "Human",
        "task": "Task",
        "summary": "Summary",
        "task+summary": "Task+\nSummary",
    }, regex=False)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^rewrite-\d+$", "Rewrite-LLM", regex=True)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^dipper-\d+$", "Dipper", regex=True)

    df['generative_model'] = df['generative_model'].replace({
        "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "human": "Human",
    }, regex=False)

    desired_order = ["Human", "Improved-\nHuman", "Rewrite-\nHuman", "Summary", "Task+\nSummary", "Task", "Rewrite-LLM",
                     "Dipper"]

    data = []
    auc_scores = []
    for detector, sub_df in df.groupby("detector"):
        for mode in desired_order[:5]:
            sub_df['is_human'] = sub_df['is_human'] | (sub_df['prompt_mode'] == mode)

            fpr, tpr, _ = get_roc_curve(sub_df, drop_intermediate=True)
            model_label = mode.replace('\n', '')

            num_human = sub_df['is_human'].sum()
            num_llm = (~sub_df['is_human']).sum()

            print(mode)
            print(num_human)
            print(num_llm)

            sub_df['weights'] = np.where(sub_df['is_human'], 1, num_human / num_llm)

            auc_scores.append({
                'detector': detector,
                'roc_auc': roc_auc_score(~sub_df['is_human'], sub_df['prediction'], sample_weight=sub_df['weights']),
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

    df_pivot = df_auc.pivot(values='roc_auc', index=['mode'], columns=['detector'])
    print(df_pivot.to_latex(header=True, index=True, float_format='%.3f'))

    grid = sns.FacetGrid(df, col="detector", hue="mode", height=2.5, aspect=1, sharex=False, sharey=False, col_wrap=2,
                         legend_out=False)

    grid.map(sns.lineplot, "fpr", "tpr", errorbar=None)

    for i, ax in enumerate(grid.axes.flatten()):
        ax.grid(True)  # Gitterlinien aktivieren
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    title_map = {
        "detect-gpt": "DetectGPT",
        "fast-detect-gpt": "FastDetectGPT",
        "ghostbuster": "Ghostbuster",
        "roberta": "RoBERTa",
        "intrinsic-dim": "IntrinsicDim",
    }
    for ax, title in zip(grid.axes.flatten(), grid.col_names):
        new_title = title_map.get(title, title)
        ax.set_title(new_title, weight='bold')

    grid.add_legend(title="Label Boundary")
    sns.move_legend(grid, "upper left", bbox_to_anchor=(.63, .26), frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(f"plots/label_boundaries_roc_all.png", dpi=300)
    plt.show()


def boundary_comparison_single(detector):
    df, _ = compute_boundary_comparison(detector)

    plt.figure(figsize=(4, 3.8))

    grid = sns.lineplot(data=df, x="fpr", y="tpr", errorbar=None, hue="mode")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.text(.53, .47, s="Random Classifier", rotation=45, horizontalalignment='center',
             verticalalignment='center', color="gray", weight='bold')

    grid.set_xlabel('False Positive Rate')
    grid.set_ylabel('True Positive Rate')

    sns.despine(offset=0, trim=True)
    plt.gca().set_aspect('equal')

    plt.tight_layout()

    plt.savefig(f"plots/label_boundaries_roc_{detector}.pdf")
    plt.show()


if "__main__" == __name__:
    boundary_comparison()
    # boundary_comparison_single(detector="detect-gpt")
