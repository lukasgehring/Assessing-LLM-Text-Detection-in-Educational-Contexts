import pandas as pd

from evaluation.utils import remove_rows_by_condition, select_best_roberta_checkpoint, get_data


def load():
    df_llama = get_data(generative_model="meta-llama/Llama-3.3-70B-Instruct", dataset=None, detector=None,
                        database="../../../database/database.db")
    df_llama.loc[df_llama['generative_model'] == 'dipper', 'generative_model'] = 'meta-llama/Llama-3.3-70B-Instruct'

    df_gpt = get_data(generative_model="gpt-4o-mini-2024-07-18", dataset=None, detector=None,
                      database="../../../database/database.db")
    df_gpt.loc[df_gpt['generative_model'] == 'dipper', 'generative_model'] = 'gpt-4o-mini-2024-07-18'
    df = pd.concat([df_llama, df_gpt]).reset_index(drop=True)

    df.drop_duplicates(inplace=True)

    df['generative_model'] = df['generative_model'].fillna("human")

    df['prompt_mode'] = df['prompt_mode'].replace({
        "improve-human": "Improved-Human",
        "rewrite-human": "Rewrite-Human",
        "human": "Human",
        "task": "Task",
        "summary": "Summary",
        "task+summary": "Task+Summary",
    }, regex=False)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^rewrite-\d+$", "Rewrite-LLM", regex=True)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^dipper-\d+$", "Dipper", regex=True)

    df['generative_model'] = df['generative_model'].replace({
        "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
        "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
        "human": "Human",
    }, regex=False)

    df = remove_rows_by_condition(df, {
        "name": "mixed",
        "prompt_mode": "task+resource",
    })

    for detector in df.detector.unique():
        if detector == "roberta":
            sub_df = []
            for dataset in df.name.unique():
                sub_df.append(select_best_roberta_checkpoint(df[df['name'] == dataset], dataset))
            df = df[df['detector'] != 'roberta']
            df = pd.concat([df, pd.concat(sub_df)])

        return df
