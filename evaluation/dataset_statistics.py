import nltk
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

from evaluation.metrics import statistical_analysis
from evaluation.utils import load_results, ExperimentConfig, load_datasets

def prettify_plot():
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

def cosine_similarity_text(row, vectorizer):
    text1 = row['human']
    text2 = row['llm']

    vectorized = vectorizer.fit_transform([text1, text2])

    similarity = cosine_similarity(vectorized[0:1], vectorized[1:2])[0][0]
    return similarity


def plot_cosine_similarity(datasets):
    vectorizer = TfidfVectorizer()

    plt.figure(figsize=(8, 6))
    for info, dataset in datasets:
        dataset["cosine_similarity"] = dataset.apply(cosine_similarity_text, axis=1, vectorizer=vectorizer)
        dataset['name'] = f"{info['info']['model']}-{info['info']['prompt_mode']}"

    df = pd.concat([i[1] for i in datasets], ignore_index=True)
    sns.kdeplot(df, x="cosine_similarity", hue="name", fill=True, alpha=.5, linewidth=0, palette="pastel")

    plt.title("Cosine Similarity Comparison", fontsize=14)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    prettify_plot()
    plt.show()

def compute_text_lengths(info_dataset_list):

    df = pd.DataFrame()

    for info, dataset in info_dataset_list:
        contained = []
        if 'name' in df.columns:
            contained = df['name'].unique()

        if f"human-{info['info']['dataset']}" not in contained:
            human_df = pd.DataFrame({'length': dataset['human'].apply(lambda x: len(x.split())), 'name': f"human-{info['info']['dataset']}"})
            df = pd.concat([df, human_df], ignore_index=True)

        if f"{info['info']['model']}-{info['info']['prompt_mode']}" not in contained:
            llm_df = pd.DataFrame({'length': dataset['llm'].apply(lambda x: len(x.split())), 'name': f"{info['info']['model']}-{info['info']['prompt_mode']}"})
            df = pd.concat([df, llm_df], ignore_index=True)
        else:
            logger.warning(f"Skipping {info['info']['model']}-{info['info']['prompt_mode']} as it is already contained in the analysis.")

    return df




def corpus_comparison(info, dataset, column_names):

    dfs = []

    for column_name in column_names:
        corpus = dataset[[column_name]].copy()
        corpus.columns = ["text"]
        corpus_stats = statistical_analysis(corpus)
        corpus_stats = corpus_stats.drop("text", axis=1)
        corpus_stats['Text Type'] = column_name

        dfs.append(corpus_stats)


    df_combined = pd.concat(dfs)
    df_melted = df_combined.melt(id_vars=['Text Type'], var_name='metric', value_name="score")

    return df_melted


def plot_text_lengths(info_dataset_list):
    df = compute_text_lengths(info_dataset_list)
    plt.figure(figsize=(8, 6))

    sns.kdeplot(df, x="length", hue="name", fill=True, alpha=.5, linewidth=0, palette="pastel")

    plt.title("Text Length (Words) Comparison", fontsize=14)
    plt.xlabel("Text Length (Words)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    prettify_plot()
    plt.show()


if "__main__" == __name__:
    config = ExperimentConfig(
        source_datasets=['brat-project'],
        # detectors=['Ghostbuster'],
        generative_models=['meta-llama/Llama-3.3-70B-Instruct'],
        # prompt_modes=['rewrite-human']
    )
    info_dataset_list = load_datasets(config)

    # plot_text_lengths(info_dataset_list)
    # plot_cosine_similarity(info_dataset_list)

    for info, dataset in info_dataset_list:
        df = corpus_comparison(info, dataset, column_names=['human', 'llm'])

        g = sns.FacetGrid(df, col='metric', col_wrap=3, sharey=False, height=4)
        g.map_dataframe(sns.violinplot, x='Text Type', y="score", hue='Text Type', palette="pastel")
        g.fig.subplots_adjust(top=0.9)
        info_dict = info['info']
        g.fig.suptitle(f"Human-written vs. LLM-generated | {info_dict['model']} | {info_dict['dataset']} | {info_dict['prompt_mode']}", fontsize=14)
        plt.show()
        break