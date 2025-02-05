import itertools
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from plot.load_results import load_results
import matplotlib.pyplot as plt

def _plot(model_names, metrics, x_metric, y_metric, colors, title, xlabel, ylabel, figsize, bounds=(0,1), num_ticks=5, legend=True, baseline=None):
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

    for i, model_name in enumerate(model_names):

        ax.plot(metrics[i][x_metric], metrics[i][y_metric], linewidth=2, color=colors[i])

        if legend:
            y_pos = .05 + (i / 20)
            x_pos = .3
            ax.plot([x_pos + .06, x_pos + .095], [y_pos, y_pos], linewidth=2, color=colors[i], alpha=.8)

            ax.text(
                x=x_pos + .11,
                y=y_pos,
                s=f"{model_name}",
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

def plot_accuracy(model_names, metrics, dataset, model, colors=mpl.colormaps['Dark2'].colors, figsize=(6,5), save_plot=False):
    fig, ax = _plot(
        model_names=model_names,
        metrics=metrics,
        x_metric="false_positive_rate",
        y_metric="accuracy",
        colors=colors,
        title=f"Accuracy on {model}-{dataset}",
        xlabel="False Positive Rate",
        ylabel="Accuracy",
        figsize=figsize,
        baseline={
            'x':[0,1],
            'y':[.5,.5],
            'label': {
                'x': .5,
                'y': .47,
                'rotation': 0,
            }
        }
    )

    if save_plot:
        plt.savefig(f"pdfs/accuracy_{model}_{dataset}.pdf", format="pdf", bbox_inches="tight")

    plt.show()

def plot_auc_curve(model_names, metrics, dataset, model, colors=mpl.colormaps['Dark2'].colors, figsize=(6,5), save_plot=False):
    fig, ax = _plot(
        model_names=model_names,
        metrics=metrics,
        x_metric="false_positive_rate",
        y_metric="true_positive_rate",
        colors=colors,
        title=f"ROC on {model}-{dataset}",
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
        plt.savefig(f"pdfs/roc_curve_{model}_{dataset}.pdf", format="pdf", bbox_inches="tight")

    plt.show()

def compute_metrics(results):

    all_metrics = []

    for result in results:
        ground_truth = [0] * len(result['predictions']['human']) + [1] * len(result['predictions']['llm'])
        y_scores = result['predictions']['human'] + result['predictions']['llm']
        false_positive_rate, true_positive_rate, thresholds = roc_curve(ground_truth, y_scores, drop_intermediate=False)

        num_thresholds = len(thresholds)

        true_positives = np.zeros(num_thresholds)
        false_negatives = np.zeros(num_thresholds)
        false_positives = np.zeros(num_thresholds)
        true_negatives = np.zeros(num_thresholds)

        for i, threshold in enumerate(thresholds):
            y_pred = [1 if score >= threshold else 0 for score in y_scores]

            true_negative, false_positive, false_negative, true_positive = confusion_matrix(ground_truth, y_pred).ravel()

            true_positives[i] = true_positive
            false_negatives[i] = false_negative
            false_positives[i] = false_positive
            true_negatives[i] = true_negative

        _epsilon = 1e-7
        accuracy = (true_negatives + true_positives) / (true_positives + true_negatives + false_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives + _epsilon)
        recall = true_positives / (true_positives + false_negatives + _epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + _epsilon)
        # TODO: ADd human and llm recall

        metrics = {
            'false_positive_rate': false_positive_rate,
            'true_positive_rate': true_positive_rate,
            'true_positive': np.array(true_positives),
            'false_negative': np.array(false_negatives),
            'false_positive': np.array(false_positives),
            'true_negative': np.array(true_negatives),
            'roc_auc': auc(false_positive_rate, true_positive_rate),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        print(f"{result['detector']}    roc_auc: {metrics['roc_auc']}")

        all_metrics.append(metrics)

    return all_metrics

def plot_predictions(results, dataset, model, colors=mpl.colormaps['Dark2'].colors, save_plot=False):
    for result in results:
        human = np.array(result['predictions']['human']).flatten()
        llm = np.array(result['predictions']['llm']).flatten()

        human_df = pd.DataFrame({'name': 'Human Text', 'score': human})
        llm_df = pd.DataFrame({'name': 'LLM Text', 'score': llm})

        df = pd.concat([human_df, llm_df])

        plt.title(f"Prediction Scores of {result['detector']} for {model}-{dataset}")
        sns.kdeplot(df, x="score", hue="name", fill=True, alpha=.5, linewidth=0, palette=colors)

        if save_plot:
            plt.savefig(f"pdfs/scores_{result['detector']}_{model}_{dataset}.pdf", format="pdf", bbox_inches="tight")
        plt.show()



def convert(r):
    if 'real' in r['predictions'].keys():
        r['predictions']['human'] = r['predictions'].pop('real')
        r['predictions']['llm'] = r['predictions'].pop('samples')

def get_results(path="../results", generative_model=None, dataset=None, drop_keys=None):
    file_names = [
        os.path.splitext(file)[0] for file in os.listdir(path) if
        os.path.isfile(os.path.join(path, file)) and os.path.splitext(file)[1] == '.gz'
    ]

    if generative_model:
        file_names = [file_name for file_name in file_names if generative_model.lower() in file_name]

    results = []
    for file_name in file_names:
        _results = load_results(file_name)


        for _result in _results:

            # skip _d version of DetectGPT and only keeping _z
            if _result['detector'] == 'DetectGPT':
                if '_d' in _result['name']:
                    continue

            if dataset:
                if dataset not in _result['info']['dataset']:
                    continue

            if drop_keys:
                for drop_key in drop_keys:
                    _result.pop(drop_key, None)

            results.append(_result)



    return results

def main():
    datasets = ['brat']
    models = ['llama-3.1-70b-instruct-131536', 'llama-3.3-70b-instruct-131538', 'gpt-4o-mini-20250130074608']
    save_plot = False

    for model, dataset in itertools.product(models, datasets):
        print(model, dataset)
        results = get_results(generative_model=model, drop_keys=['raw_results'], dataset=dataset)

        plot_predictions(results, dataset=dataset, model=model, save_plot=save_plot)

        metrics = compute_metrics(results=results)

        labels = [result['detector'] for result in results]
        plot_auc_curve(model_names=labels, metrics=metrics, dataset=dataset, model=model, save_plot=save_plot)
        plot_accuracy(model_names=labels, metrics=metrics, dataset=dataset, model=model, save_plot=save_plot)

if __name__ == "__main__":
    main()

