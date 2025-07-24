import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from joypy import joyplot
from matplotlib import pyplot as plt, ticker, cm

from database.interface import get_predictions
from evaluation.utils import get_data, remove_rows_by_condition, remove_job_id_from_prompt_mode


def seaborn_ridge(generative_model: str, dataset: str, detector: str, save_plot: bool = False):
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    df = get_predictions(detector=detector, dataset=dataset)
    df = remove_rows_by_condition(df, {
        'name': "mixed"
    })
    remove_job_id_from_prompt_mode(df)

    df['prompt_mode'] = df['prompt_mode'].replace({
        "improve-human": "Improve-Human",
        "rewrite-human": "Rewrite-Human",
        "human": "Human",
        "task": "Task",
        "summary": "Summary",
        "task+summary": "Task+Summary",
        "rewrite-llm": "Rewrite-LLM",
        "dipper": "Dipper",
    }, regex=False)

    desired_order = ["Dipper", "Rewrite-LLM", "Task", "Task+Summary", "Summary", "Rewrite-Human", "Improve-Human",
                     "Human"]
    df["prompt_mode"] = pd.Categorical(df["prompt_mode"], categories=desired_order, ordered=True)

    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row="prompt_mode", hue="prompt_mode", aspect=12, height=.5, palette="viridis")

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "prediction",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "prediction", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=1.5, linestyle="--", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.1, .2, label, fontweight="bold", color="black",
                ha="right", va="center", transform=ax.transAxes)

    g.map(label, "prediction")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.3)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="")
    g.despine(bottom=True, left=True)
    plt.subplots_adjust(bottom=0.1, top=1, left=0.35, right=0.98)

    if save_plot:
        plt.savefig(f"plots/ridgeline_{detector}_{dataset}.pdf", format="pdf",
                    transparent=False)

    plt.show()


if __name__ == "__main__":
    detect_gpt = "detect-gpt"
    intrinsic_dim = "intrinsic-dim"
    roberta = "roberta"
    ghostbuster = "ghostbuster"

    llama = "meta-llama/Llama-3.3-70B-Instruct"
    gpt = "gpt-4o-mini-2024-07-18"

    aae = "argument-annotated-essays"
    bawe = "BAWE"
    persuade = "persuade"

    # two possible plots
    seaborn_ridge(generative_model=gpt, dataset=None, detector=detect_gpt, save_plot=True)

    # ridgeline_plot(
    #    detector="detect-gpt",
    #    save_plot=True,
    #    # dataset="argument-annotated-essays",
    #    dataset="persuade",
    #    bins=25,
    #    rewrite_dipper_id=143932
    # )
