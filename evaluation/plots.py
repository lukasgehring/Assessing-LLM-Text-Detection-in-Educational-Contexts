import itertools
import os
import sys
from typing import Union, List

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import ndarray

from evaluation.metrics import compute_metrics
from evaluation.utils import ExperimentConfig, load_results

def highlighted_scatter_plot(data: DataFrame, x_column:str, y_column:str, highlighted_indices: Union[List[int],ndarray[int], slice] = None, title:str=None) -> None:
    """
    Plot a scatter plot of data with highlighted indices.

    :param data: DataFrame containing data
    :param x_column: column name of x-axis
    :param y_column: column name of y-axis
    :param highlighted_indices: indices to highlight
    :param title: title of plot
    """

    # plot data
    sns.scatterplot(data=data, x=x_column, y=y_column, color='lightgray')

    # plot highlights
    if highlighted_indices is not None:
        sns.scatterplot(data=data.loc[highlighted_indices], x=x_column, y=y_column, color='red', label='strange samples', s=100)

    # setup
    if title is not None:
        plt.title(label=title, fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.show()


def _plot_multi(results, identifier, x_metric, y_metric, colors, title, xlabel, ylabel, figsize, bounds=(0,1), num_ticks=5, legend=True, baseline=None):
    fig, ax = plt.subplots(figsize=figsize)

    if baseline:
        # add baseline
        ax.plot(baseline['x'], baseline['y'], linestyle="--", color='lightgray', linewidth=2)

        ax.text(
            x=baseline['label']['x'],
            y=baseline['label']['y'],
            s=f"Random classifier",
            rotation=baseline['label']['rotation'],
            fontweight="bold",
            color='lightgray',
            horizontalalignment="center",
            verticalalignment="center",
        )

    for i, (id, x, y) in enumerate(sorted(zip(results[identifier], results[x_metric], results[y_metric]))):
        ax.plot(x, y, linewidth=2, color=colors[i])

        if legend:
            y_pos = .05 + (i / 20)
            x_pos = .3
            ax.plot([x_pos + .06, x_pos + .095], [y_pos, y_pos], linewidth=2, color=colors[i], alpha=.8)

            ax.text(
                x=x_pos + .11,
                y=y_pos,
                s=f"{id}",
                fontweight="bold",
                color=colors[i],
                horizontalalignment="left",
                verticalalignment="center",
            )

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    ax.set_aspect('equal', adjustable='box')

    ax.spines[['right', 'top']].set_visible(False)
    ax.spines[['left', 'bottom']].set_bounds(bounds[0], bounds[1])
    ax.spines[['left', 'bottom']].set_position('zero')

    ticks = np.linspace(bounds[0], bounds[1], num_ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    ax.set_yticks(ticks)
    ax.set_yticklabels([""] + ticks[1:].tolist())

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    fig.tight_layout()

    return fig, ax

def _plot_single(results, x_metric, y_metric, title, xlabel, ylabel, figsize, bounds=(0,1), num_ticks=5, legend=True, baseline=None):
    fig, ax = plt.subplots(figsize=figsize)

    df = results.sort_values(by=[x_metric])

    if baseline:
        # add baseline
        ax.plot(baseline['x'], baseline['y'], linestyle="--", color='lightgray', linewidth=2)

        if 'label' in baseline.keys():
            ax.text(
                x=baseline['label']['x'],
                y=baseline['label']['y'],
                s=f"Random classifier",
                rotation=baseline['label']['rotation'],
                fontweight="bold",
                color='lightgray',
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.plot(df[x_metric], df[y_metric], linewidth=2)

    legend=False
    if legend:
        y_pos = .05 + (0 / 20)
        x_pos = .3
        ax.plot([x_pos + .06, x_pos + .095], [y_pos, y_pos], linewidth=2, color=colors[i], alpha=.8)

        ax.text(
            x=x_pos + .11,
            y=y_pos,
            s=f"{id}",
            fontweight="bold",
            color=colors[i],
            horizontalalignment="left",
            verticalalignment="center",
        )

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    #ax.set_aspect('equal', adjustable='box')

    ax.spines[['right', 'top']].set_visible(False)
    #ax.spines[['left', 'bottom']].set_bounds(bounds[0], bounds[1])
    #ax.spines[['left', 'bottom']].set_position('zero')

    ticks = np.linspace(bounds[0], bounds[1], num_ticks)
    #ax.set_xticks(ticks)
    #ax.set_xticklabels(ticks)

    #ax.set_yticks(ticks)
    #ax.set_yticklabels([""] + ticks[1:].tolist())

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    fig.tight_layout()

    return fig, ax

def plot_auc_curve(results, identifier, title, colors=mpl.colormaps['Dark2'].colors, figsize=(6,5), save_plot=False):
    fig, ax = _plot_multi(
        results=results,
        identifier=identifier,
        x_metric="false_positive_rate",
        y_metric="true_positive_rate",
        colors=colors,
        title=title,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        figsize=figsize,
        baseline={
            'x': [0, 1],
            'y': [0, 1],
            'label': {
                'x': .53,
                'y': .47,
                'rotation': 45,
            }
        }
    )

    if save_plot:
        plt.savefig(f"plots/roc_curve_{title}.pdf", format="pdf", bbox_inches="tight", transparent=True)

    plt.show()



def plot_predictions(info_result_list, colors=mpl.colormaps['Dark2'].colors, save_plot=False):
    for info, result in info_result_list:
        human = np.array(result['predictions']['human']).flatten()

        llm = np.array(result['predictions']['llm']).flatten()

        human_df = pd.DataFrame({'name': 'Human Text', 'score': human})
        llm_df = pd.DataFrame({'name': 'LLM Text', 'score': llm})

        df = pd.concat([human_df, llm_df])

        plt.rcParams["figure.figsize"] = 10.5, 7
        sns.set_theme(style="whitegrid")
        plt.title(f"Prediction Scores of {info['model']} for {info['dataset']['info']['model']}-{info['dataset']['info']['dataset']}")

        sns.kdeplot(df,
                    x="score",
                    hue="name",
                    fill=True,
                    alpha=.4,
                    palette="viridis",
                    common_norm=False
                    )

        if save_plot:
            plt.savefig(f"pdfs/scores_{info['model']}_{info['dataset']['info']['model']}_{info['dataset']['info']['dataset']}.pdf", format="pdf", bbox_inches="tight")

        plt.show()



def convert(r):
    if 'real' in r['predictions'].keys():
        r['predictions']['human'] = r['predictions'].pop('real')
        r['predictions']['llm'] = r['predictions'].pop('samples')




if __name__ == "__main__":
    detectors = ['IntrinsicDim (PHD)', 'Ghostbuster', 'DetectGPT']
    generative_models = ['meta-llama/Llama-3.3-70B-Instruct', 'gpt-4o-mini-2024-07-18']

    config = ExperimentConfig(
        source_datasets=['brat-project'],
        detectors=['RoBERTa'],
        generative_models=[generative_models[0]],
        prompt_modes=['task']
    )

    info_result_list = load_results(config) #, drop_keys=['raw_results'])
    results = compute_metrics(info_result_list=info_result_list)

    print(results)
    roc_auc_inf = results.loc[results["max_words"].isna(), "roc_auc"].values[0]
    _plot_single(results=results,
                 x_metric="max_words",
                 y_metric="roc_auc",
                 title="Test",
                 xlabel="Test",
                 ylabel="Test",
                 figsize=(5,5),
                 baseline={
                     'x': [100, 250],
                     'y': [roc_auc_inf, roc_auc_inf]
                 }
                 )

    plt.show()