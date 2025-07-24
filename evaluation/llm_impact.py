from itertools import product

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

import seaborn as sns

from database.interface import get_predictions
from evaluation.utils import remove_rows_by_condition, get_data, select_best_roberta_checkpoint, \
    map_dipper_to_generative_model, set_label, remove_job_id_from_prompt_mode, filter_and_get_roc_auc, \
    get_roc_auc, highlight_max

sns.set_theme(context="paper", style=None, font_scale=1, rc={
    # grid
    "grid.linewidth": .5,

    # legend
    'legend.handletextpad': .5,
    'legend.handlelength': 1.0,
    # 'legend.labelspacing': 0.5,
    # "legend.loc": "upper center",

    # axes
    'axes.spines.right': False,
    'axes.spines.top': False,

    # yticks
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.4,

    # save
    'savefig.format': 'pdf'
})


def prompt_mode_performance():
    df = get_predictions(max_words=-1)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource"
    })

    # make sure, to load the correct dipper results
    df = df.apply(map_dipper_to_generative_model, axis=1)

    # set human label of improve and rewrite-human
    set_label(df)

    # rewrite job_id from rewrite-llm and dipper
    remove_job_id_from_prompt_mode(df)

    results = []
    for generative_model in ['gpt-4o-mini-2024-07-18', 'meta-llama/Llama-3.3-70B-Instruct']:
        gen_df = df[df['generative_model'].isin([generative_model, 'human'])]
        for (detector,), sub_df in gen_df.groupby(['detector']):
            for prompt_mode in sub_df.prompt_mode.unique():
                roc_auc = filter_and_get_roc_auc(sub_df, prompt_mode)
                if roc_auc is not None:
                    results.append({
                        "detector": detector,
                        "generative_model": generative_model,
                        "prompt_mode": prompt_mode,
                        "roc_auc": roc_auc
                    })
            results.append({
                "detector": detector,
                "generative_model": generative_model,
                "prompt_mode": "all",
                "roc_auc": get_roc_auc(sub_df)
            })
    df = pd.DataFrame(results)

    def combine_roc(df):
        # Aufteilen in GPT-Modelle und LLaMA-Modelle
        gpt_df = df[df['generative_model'].str.contains("gpt", case=False)].copy()
        llama_df = df[df['generative_model'].str.contains("llama", case=False)].copy()

        # Relevante Spalten umbenennen
        gpt_df = gpt_df[['detector', 'prompt_mode', 'roc_auc']].rename(columns={'roc_auc': 'roc_auc_gpt'})
        llama_df = llama_df[['detector', 'prompt_mode', 'roc_auc']].rename(columns={'roc_auc': 'roc_auc_llama'})

        # Mergen nach detector + prompt_mode
        merged = pd.merge(gpt_df, llama_df, on=['detector', 'prompt_mode'], how='inner')

        # Kombinieren der ROC-Werte als String
        merged['roc_auc'] = merged['roc_auc_llama'] - merged['roc_auc_gpt']

        # Nur gewünschte Spalten behalten
        final_df = merged[['detector', 'prompt_mode', 'roc_auc']]
        return final_df

    df = combine_roc(df)

    print(df)
    df_wide = df.pivot(index='detector', columns='prompt_mode', values='roc_auc').reset_index()
    df_wide = df_wide.set_index('detector')
    order = ["improve-human", "rewrite-human", "summary", "task+summary", "task", "rewrite-llm", "dipper", "all"]
    df_wide = df_wide[order]

    latex_table = highlight_max(df_wide, add_sign=True).to_latex(
        index=True,
        header=True,
        column_format="lcccccccc",
        escape=False
    )

    print(latex_table)


def generative_model_comparison():
    df = get_predictions(max_words=-1)
    df = select_best_roberta_checkpoint(df)
    df = remove_rows_by_condition(df, conditions={
        "prompt_mode": "task+resource"
    })

    # make sure, to load the correct dipper results
    df = df.apply(map_dipper_to_generative_model, axis=1)

    # set human label of improve and rewrite-human
    set_label(df)

    # rewrite job_id from rewrite-llm and dipper
    remove_job_id_from_prompt_mode(df)

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
    print(pivot_df.columns)
    print(pivot_df['meta-llama/Llama-3.3-70B-Instruct'] - pivot_df['gpt-4o-mini-2024-07-18'])
    print((pivot_df['meta-llama/Llama-3.3-70B-Instruct'] - pivot_df['gpt-4o-mini-2024-07-18']).mean())


if __name__ == '__main__':
    # prompt_mode_performance()
    generative_model_comparison()
