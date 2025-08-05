import math
import sqlite3

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database.interface import get_answers

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # lines
    "lines.linewidth": 2,

    # grid
    "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    'legend.labelspacing': 0.5,
    # "legend.loc": "upper center",

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.4,

    # save
    'savefig.format': 'pdf'
})


def calculate_cosine_similarity(original, rewrite1, rewrite2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original, rewrite1, rewrite2])

    cos_sim_1 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    cos_sim_2 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])[0][0]

    return cos_sim_1, cos_sim_2


def rewrite_cosine_similarity_multi(prompt_mode_1, prompt_mode_2):
    dfs = []

    for model in ["meta-llama/Llama-3.3-70B-Instruct", "gpt-4o-mini-2024-07-18"]:
        for dataset in ['argument-annotated-essays', 'persuade', 'BAWE']:
            human = get_answers(database="../../database/database.db", dataset=dataset, is_human=True)
            rewrite = get_answers(database="../../database/database.db", dataset=dataset, is_human=False,
                                  prompt_mode=prompt_mode_1, generative_model=model)
            improve = get_answers(database="../../database/database.db", dataset=dataset, is_human=False,
                                  prompt_mode=prompt_mode_2, generative_model=model)

            if "summary" in prompt_mode_1 and "summary" in prompt_mode_2:
                with sqlite3.connect("../../database/database.db") as connection:
                    summaries = pd.read_sql("select * from summaries", connection)

                human = human[human['id'].isin(summaries['source_id'])]
            elif "summary" in prompt_mode_1:
                with sqlite3.connect("../../database/database.db") as connection:
                    summaries = pd.read_sql("select * from summaries", connection)

                human = human[human['id'].isin(summaries['source_id'])]
                improve = improve[improve.index.isin(human.index)]
            elif "summary" in prompt_mode_2:
                with sqlite3.connect("../../database/database.db") as connection:
                    summaries = pd.read_sql("select * from summaries", connection)

                human = human[human['id'].isin(summaries['source_id'])]
                rewrite = rewrite[rewrite.index.isin(human.index)]

            human.reset_index(drop=True, inplace=True)
            rewrite.reset_index(drop=True, inplace=True)
            improve.reset_index(drop=True, inplace=True)

            human['prompt_mode'] = 'human'
            rewrite['prompt_mode'] = prompt_mode_1
            improve['prompt_mode'] = prompt_mode_2

            human['cos_sim_rewrite'], human['cos_sim_improve'] = zip(*human.apply(
                lambda row: calculate_cosine_similarity(row['answer'], rewrite.loc[row.name, 'answer'],
                                                        improve.loc[row.name, 'answer']),
                axis=1
            ))

            df = human[['cos_sim_rewrite', 'cos_sim_improve']].copy()

            df['generative_model'] = model
            df['dataset'] = dataset

            dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)

    min_1 = df['cos_sim_rewrite'].min()
    min_2 = df['cos_sim_improve'].min()

    min_total = min_1 if min_1 < min_2 else min_2
    min_floor = math.floor(min_total * 100) / 100.0

    _, bins = np.histogram([min_floor, 1], bins=20)

    grid = sns.FacetGrid(df, col="dataset", row="generative_model", sharex=True, sharey=False, height=2, aspect=1.3)

    grid.map_dataframe(sns.histplot, x="cos_sim_rewrite", bins=bins, kde=False, alpha=0.8, label=prompt_mode_1,
                       color=sns.color_palette()[0], edgecolor="w")
    grid.map_dataframe(sns.histplot, x="cos_sim_improve", bins=bins, kde=False, alpha=0.8, label=prompt_mode_2,
                       color=sns.color_palette()[1], edgecolor="w")

    grid.set_xlabels("")

    titles = ["Argument-Annotated-Essays", "PERSUADE", "BAWE", "", "", ""]
    xlabels = ["", "", "", "", "Cosine Similarity (Between Human-Texts and Rewrites)", ""]
    ylabels = ["llama-3.3-70b", "", "", "gpt-4-o-mini", "", ""]
    for ax, title, xlabel, ylabel in zip(grid.axes.flatten(), titles, xlabels, ylabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # grid.add_legend(ncol=1)
    # plt.legend(title="Modell", loc="lower right", bbox_to_anchor=(0, -0.65), ncol=2)

    sns.despine(offset=0, trim=True)
    handles, labels = grid.axes[0, 0].get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(.5, .95), frameon=False)
    grid.tight_layout()
    plt.subplots_adjust(top=0.85)
    for ax in grid.axes.flatten():
        ax.grid()
    plt.savefig(f"plots/rewrite_cos_sim_{prompt_mode_1}_{prompt_mode_2}.pdf")

    plt.show()


def rewrite_cosine_similarity():
    human_dfs = []
    rewrite_dfs = []
    improve_dfs = []

    for model in ["meta-llama/Llama-3.3-70B-Instruct", "gpt-4o-mini-2024-07-18"]:
        for dataset in ['argument-annotated-essays', 'persuade', 'BAWE']:
            human_dfs.append(get_answers(database="../../database/database.db", dataset=dataset, is_human=True))
            rewrite_dfs.append(get_answers(database="../../database/database.db", dataset=dataset, is_human=False,
                                           prompt_mode="rewrite-human", generative_model=model))
            improve_dfs.append(get_answers(database="../../database/database.db", dataset=dataset, is_human=False,
                                           prompt_mode="improve-human", generative_model=model))

    human = pd.concat(human_dfs).reset_index(drop=True)
    rewrite = pd.concat(rewrite_dfs).reset_index(drop=True)
    improve = pd.concat(improve_dfs).reset_index(drop=True)

    human['cos_sim_rewrite'], human['cos_sim_improve'] = zip(*human.apply(
        lambda row: calculate_cosine_similarity(row['answer'], rewrite.loc[row.name, 'answer'],
                                                improve.loc[row.name, 'answer']),
        axis=1
    ))

    plt.figure(figsize=(4, 2.5))

    min_1 = human['cos_sim_rewrite'].min()
    min_2 = human['cos_sim_improve'].min()

    min_total = min_1 if min_1 < min_2 else min_2
    min_total = math.floor(min_total * 100) / 100.0

    _, bins = np.histogram([min_total, 1], bins=20)

    sns.histplot(data=human, x="cos_sim_rewrite", bins=bins, kde=False, label=fr"Rewrite", alpha=0.8, edgecolor="w")
    sns.histplot(data=human, x="cos_sim_improve", bins=bins, kde=False, label=fr"Improve", alpha=0.8, edgecolor="w")
    plt.legend()
    plt.ylabel("Frequency", weight="bold")
    plt.xlabel("Cosine Similarity", weight="bold")
    sns.despine(offset=0, trim=True)
    print("rewrite: mean", human.cos_sim_rewrite.mean(), "std", human.cos_sim_rewrite.std())
    print("improve: mean", human.cos_sim_improve.mean(), "std", human.cos_sim_improve.std())

    plt.tight_layout()

    plt.savefig("plots/rewrite_cos_sim.pdf", format="pdf", bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    rewrite_cosine_similarity()
    rewrite_cosine_similarity_multi(prompt_mode_1="rewrite-human", prompt_mode_2="improve-human")
