import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.colors as mcolors
from database.interface import get_predictions
from evaluation.metrics import get_roc_curve
from evaluation.utils import get_data, remove_rows_by_condition, select_best_roberta_checkpoint, \
    map_dipper_to_generative_model, highlight_max

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


def pastel_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([x + (1 - x) * amount for x in c])


custom_palette = ["#E69F00", "#0072B2", "#009E73", "#D55E00"]
pastel_colors = [pastel_color(c, 0.2) for c in custom_palette]
sns.set_palette(pastel_colors)


def prediction_boxplot(dataset: str, detector: str, function=roc_curve):
    df = get_predictions(max_words=-1, detector=detector, dataset=dataset)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource"
    })

    df = df.apply(map_dipper_to_generative_model, axis=1)

    df['generative_model'] = df['generative_model'].fillna("human")

    df['prompt_mode'] = df['prompt_mode'].replace({
        "improve-human": "Improve-\nHuman",
        "rewrite-human": "Rewrite-\nHuman",
        "human": "Human",
        "task": "Task",
        "summary": "Summary",
        "task+summary": "Task+\nSummary",
    }, regex=False)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^rewrite-\d+$", "Rewrite-\nLLM", regex=True)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^dipper-\d+$", "Humanize", regex=True)

    seperate_gen_models = False
    if seperate_gen_models:
        df['generative_model'] = df['generative_model'].replace({
            "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
            "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
            "human": "Human",
        }, regex=False)
    else:
        df['generative_model'] = df['generative_model'].replace({
            "meta-llama/Llama-3.3-70B-Instruct": "LLM",
            "gpt-4o-mini-2024-07-18": "LLM",
            "human": "Human",
        }, regex=False)

    desired_order = ["Human", "Improve-\nHuman", "Rewrite-\nHuman", "Summary", "Task+\nSummary", "Task",
                     "Rewrite-\nLLM",
                     "Humanize"]

    plt.figure(figsize=(10, 6) if seperate_gen_models else (10, 6.5))
    # can be changed to boxplot
    sns.boxenplot(df,
                  x="prompt_mode",
                  y="prediction",
                  legend=False,
                  order=desired_order,
                  fill=False,
                  width=0.1 if seperate_gen_models else 0.2,
                  gap=4,
                  linewidth=3,
                  zorder=2,
                  color=pastel_colors[1]
                  )
    # plt.legend(title="Text Author")
    plt.ylabel("Score", fontweight='bold')
    plt.xlabel("Contribution Level", fontweight='bold')
    plt.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25, zorder=0)
    # plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)
    plt.grid(True, which='minor', linestyle='-', linewidth=0.2, alpha=0.15, zorder=0)
    # sns.despine(offset=0, trim=True)
    plt.xticks(rotation=0, ha='center', fontsize=17, style='italic')

    plt.tight_layout(rect=(0, 0., 0.93, 1))
    plt.ylim(-2, 2.5)
    plt.savefig(f"plots/prompt_mode_{detector}.pdf")

    # plt.show()


if __name__ == "__main__":
    prediction_boxplot(dataset=None, detector="detect-gpt")
