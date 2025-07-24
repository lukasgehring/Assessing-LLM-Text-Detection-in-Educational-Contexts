import sys

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from database.interface import get_answers, get_human_meta_data
from evaluation.metrics import statistical_analysis

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # lines
    "lines.linewidth": 2,

    # grid
    "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    # 'legend.labelspacing': 0.5,
    # "legend.loc": "upper center",

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.4,

    # save
    'savefig.format': 'pdf'
})


def prettify_plot():
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def cosine_similarity_text(row, vectorizer):
    text1 = row['human']
    text2 = row['llm']

    vectorized = vectorizer.fit_transform([text1, text2])

    similarity = cosine_similarity(vectorized[0:1], vectorized[1:2])[0][0]
    return similarity


def compute_text_lengths(info_dataset_list):
    df = pd.DataFrame()

    for info, dataset in info_dataset_list:
        contained = []
        if 'name' in df.columns:
            contained = df['name'].unique()

        if f"human-{info['info']['dataset']}" not in contained:
            human_df = pd.DataFrame({'length': dataset['human'].apply(lambda x: len(x.split())),
                                     'name': f"human-{info['info']['dataset']}"})
            df = pd.concat([df, human_df], ignore_index=True)

        if f"{info['info']['model']}-{info['info']['prompt_mode']}" not in contained:
            llm_df = pd.DataFrame({'length': dataset['llm'].apply(lambda x: len(x.split())),
                                   'name': f"{info['info']['model']}-{info['info']['prompt_mode']}"})
            df = pd.concat([df, llm_df], ignore_index=True)
        else:
            logger.warning(
                f"Skipping {info['info']['model']}-{info['info']['prompt_mode']} as it is already contained in the analysis.")

    return df


def corpus_comparison(dataset):
    dfs = []
    corpus_stats = statistical_analysis(dataset[dataset.is_human == 1][["answer"]], column="answer")
    corpus_stats = corpus_stats.drop("answer", axis=1)
    corpus_stats['Text Type'] = "Human"
    dfs.append(corpus_stats)
    corpus_stats = statistical_analysis(dataset[dataset.is_human == 0][["answer"]], column="answer")
    corpus_stats = corpus_stats.drop("answer", axis=1)
    corpus_stats['Text Type'] = "LLM"
    dfs.append(corpus_stats)

    df_combined = pd.concat(dfs)
    df_melted = df_combined.melt(id_vars=['Text Type'], var_name='metric', value_name="score")
    return df_melted


def multiplot_text_statistics(database="../../database/database.db", dataset="BAWE",
                              generative_model="meta-llama/Llama-3.3-70B-Instruct", prompt_mode="task",
                              save_plot: bool = False):
    df = get_answers(
        database=database,
        dataset=dataset,
        is_human=True,
        generative_model=generative_model,
        prompt_mode=prompt_mode,
    )

    df = pd.concat([
        df,
        get_answers(
            database=database,
            dataset=dataset,
            is_human=False,
            generative_model=generative_model,
            prompt_mode=prompt_mode
        )
    ]).reset_index(drop=True)

    result_df = corpus_comparison(df)

    plt.figure(figsize=(8, 5))

    for metric in result_df['metric'].unique():
        plt.subplot(2, 3, list(result_df['metric'].unique()).index(metric) + 1)
        sns.histplot(result_df[result_df['metric'] == metric], x='score', hue='Text Type', kde=True, bins=10)
        plt.title(f"{metric.replace('_', ' ').title()}")
        prettify_plot()

    plt.tight_layout()

    if save_plot:
        plt.savefig(f"plots/dataset_stats_{dataset}_{generative_model.split('/')[-1]}_{prompt_mode}.pdf")
    plt.show()


def human_answer_statistics(database="../../database/database.db", dataset="BAWE",
                            save_plot: bool = False):
    df = get_answers(
        database=database,
        dataset=dataset,
        is_human=True
    )

    df = get_human_meta_data(database="../../database/database.db", answer_ids=df.id)

    columns_to_plot = ["level", "grade", "gender", "L1"]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for i, col in enumerate(columns_to_plot):
        counts = df[col].value_counts()
        threshold = 0.1 * counts.sum()

        counts_filtered = counts.copy()
        counts_filtered[counts < threshold] = 0
        counts_filtered['Other'] = counts[counts < threshold].sum()
        counts_filtered = counts_filtered[counts_filtered > 0]

        counts_filtered.plot(
            kind='pie',
            ax=axes[i],
            autopct='%1.1f%%',
            shadow=False,
            title=col.title(),
            legend=False,
            wedgeprops=dict(edgecolor='w'),
        )
        axes[i].set_ylabel("")

    plt.tight_layout()

    if save_plot:
        plt.savefig(f"plots/human_meta_{dataset}.pdf", format="pdf",
                    bbox_inches="tight",
                    transparent=False)

    plt.show()


if "__main__" == __name__:
    llama = "meta-llama/Llama-3.3-70B-Instruct"
    gpt = "gpt-4o-mini-2024-07-18"

    aae = "argument-annotated-essays"
    bawe = "BAWE"
    persuade = "persuade"

    multiplot_text_statistics(generative_model=gpt, dataset=aae, prompt_mode="task", save_plot=True)
    multiplot_text_statistics(generative_model=gpt, dataset=bawe, prompt_mode="task", save_plot=True)
    multiplot_text_statistics(generative_model=gpt, dataset=persuade, prompt_mode="task", save_plot=True)

    multiplot_text_statistics(generative_model=llama, dataset=aae, prompt_mode="task", save_plot=True)
    multiplot_text_statistics(generative_model=llama, dataset=bawe, prompt_mode="task", save_plot=True)
    multiplot_text_statistics(generative_model=llama, dataset=persuade, prompt_mode="task", save_plot=True)
    # human_answer_statistics(save_plot=True)
