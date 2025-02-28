import itertools

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt

from evaluation.metrics import compute_metrics
from evaluation.plots import plot_auc_curve, plot_predictions
from evaluation.threshold import get_threshold
from evaluation.utils import ExperimentConfig, load_results




def per_model_task_comparison():
    models = ['DetectGPT', 'Ghostbuster', 'RoBERTa']
    llms = ['meta-llama/Llama-3.3-70B-Instruct', 'gpt-4o-mini-2024-07-18']
    datasets = ['brat-project', 'ielts']
    # datasets = ['ielts']

    for model, llm, dataset in itertools.product(models, llms, datasets):
        print(model, llm, dataset)
        config = ExperimentConfig(
            detectors=[model],
            generative_models=[llm],
            source_datasets=[dataset],
            max_words=[None],
            attack="dipper"
        )
        results = load_results(config)

        if not results:
            continue

        #plot_predictions(results)

        # break

        results = compute_metrics(info_result_list=results)

        results['prompt_mode'] = results['prompt_mode'].replace(r"^rewrite-\d+", "rewrite-llm", regex=True)

        plot_auc_curve(results, identifier="prompt_mode", title=f"ROC {model} on {llm.split('/')[-1]} | {dataset}", save_plot=True)


def compute_all_results():
    models = ['DetectGPT', 'Ghostbuster', 'RoBERTa']
    llms = ['meta-llama/Llama-3.3-70B-Instruct', 'gpt-4o-mini-2024-07-18']

    for model, llm in itertools.product(models, llms):
        print(model, llm)

def kde_max(text):
    kde = gaussian_kde(text)
    x_values = np.linspace(text.min(), text.max(), 100)
    density_values = kde(x_values)
    max_density_index = np.argmax(density_values)
    max_density_x = x_values[max_density_index]
    max_density_y = density_values[max_density_index]
    return max_density_x, max_density_y

def rename_df(df):
    df['prompt_mode'] = df['prompt_mode'].replace("improve-human", "Improved-Human")
    df['prompt_mode'] = df['prompt_mode'].replace(r"^rewrite-\d+$", "Rewrite-LLM", regex=True)
    df['prompt_mode'] = df['prompt_mode'].replace("rewrite-human", "Rewrite-Human")
    df['prompt_mode'] = df['prompt_mode'].replace("task", "Task")
    df['prompt_mode'] = df['prompt_mode'].replace("summary", "Summary")
    df['prompt_mode'] = df['prompt_mode'].replace("task+summary", "Task+Summary")

def test(save_plot=False):
    config = ExperimentConfig(
        detectors=["DetectGPT"],
        generative_models=["meta-llama/Llama-3.3-70B-Instruct"],
        # generative_models=["gpt-4o-mini-2024-07-18"],
        source_datasets=["brat-project"],
        max_words=[None],
        attack=None
    )

    color1 = '#84DBFA'
    color11 = '#558CA0'
    color2 = '#FCAF3E'
    color21 = '#CE5C00'
    color3 = '#2E3436'

    thresholds = get_threshold(config)
    rename_df(thresholds)
    thresholds.sort_values(by=['prompt_mode'], inplace=True, ignore_index=True)

    results = load_results(config)
    dfs = []
    for info, result in results:
        data = result['predictions']
        data['prompt_mode'] = info['dataset']['info']['prompt_mode']

        if data['prompt_mode'] == 'task':
            human_predictions = data.pop("human")
        elif data['prompt_mode'] == 'improve-human':
            del data["llm"]
        else:
            del data['human']

        dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, axis=0, ignore_index=True)

    rename_df(df)

    data = pd.melt(df, id_vars=['prompt_mode']).dropna().reset_index(drop=True).sort_values(by = ['prompt_mode', 'variable'])

    names = data['prompt_mode'].unique()

    annot = pd.DataFrame({
        'xy': data.groupby(["prompt_mode", "variable"])["value"].apply(kde_max).values,
        'y_offset': [2] * len(names),
        'text': names
    })

    baseline_annot = pd.DataFrame({
        'xy': [kde_max(pd.Series(human_predictions))],
        'y_offset': [2],
        'text': ["Human"]
    })

    g = sns.FacetGrid(data, col='prompt_mode', hue='prompt_mode', col_wrap=3)
    g.set_titles("")
    g.set(ylim=(0, 30))
    sns.set_style("white")

    # Poster hat eine Column Breite von 14.3 inch
    # g.fig.set_size_inches(14.3, 10)
    alpha = .8
    for point, ax in enumerate(g.axes.flat):
        sns.kdeplot(human_predictions, ax=ax, color=color1, linewidth=2, fill=True, alpha=alpha, common_norm=False)
        #ax.plot([annot.xy[point][0], annot.xy[point][0]], [0, annot.xy[point][1]], color=color21, linestyle='dashed', linewidth=2)
        ax.axvline(thresholds['optimal_threshold'][point], color=color3, linestyle='dashed', linewidth=2)
        #ax.plot([baseline_annot.xy[0][0], baseline_annot.xy[0][0]], [0, baseline_annot.xy[0][1]], color=color11, linestyle='dashed', linewidth=2)
    g = g.map(sns.kdeplot, "value", cut=0,fill=True, common_norm=False, alpha=alpha, linewidth=2, color=color2)


    for point, ax in enumerate(g.axes.flat):
        ax.text(annot.xy[point][0], annot.xy[point][1] + annot.y_offset[point], annot.text[point], horizontalalignment='center', size='medium', backgroundcolor="white")
        ax.text(baseline_annot.xy[0][0], baseline_annot.xy[0][1] + baseline_annot.y_offset[0], baseline_annot.text[0], horizontalalignment='center', size='medium', backgroundcolor="white")

    if save_plot:
        plt.savefig(f"plots/prediction_distribution_{config.detectors[0]}_{config.generative_models[0].split('/')[-1]}_{config.source_datasets[0]}.pdf", format="pdf", bbox_inches="tight", transparent=True)

    plt.show()



if __name__ == '__main__':
    # per_model_task_comparison()
    test(save_plot=True)