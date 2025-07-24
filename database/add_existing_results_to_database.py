import datetime
import hashlib
import sqlite3
import sys

import nltk
import numpy as np
import pandas as pd
import transformers

from database.interface import add_experiment, add_predictions, get_answers, get_dataset
from evaluation.utils import load_results, ExperimentConfig

"""
This code adds most of the results to the database.
Note that 3 ghostbuster and 14 detect-gpt experiments needs to be re-run as the predictions could not be mapped to the correct answer!
"""


def detect_gpt_preprocess(data):
    original = data["human"]
    generated = data["llm"]

    # remove whitespaces
    original = original.str.strip()
    generated = generated.str.strip()

    # remove newlines
    original = original.apply(lambda x: ' '.join(x.split()))
    generated = generated.apply(lambda x: ' '.join(x.split()))

    original = original.to_list()
    generated = generated.to_list()

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-3b", cache_dir="../.resources")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_original = tokenizer(original)
    tokenized_generated = tokenizer(generated)

    mask = [len(x) <= 512 and len(y) <= 512 for x, y in
            zip(tokenized_original["input_ids"], tokenized_generated["input_ids"])]
    original = [value for i, value in enumerate(original) if mask[i]]
    generated = [value for i, value in enumerate(generated) if mask[i]]

    return data[mask]


def process_text(text):
    return text.replace("\n", "")  # TODO: Add whitespace when replacing


def process_llm_text(text):
    text = text.split("Note:")[0]

    if text.split("\n")[0][-1] == ":":
        text = "\n".join(text.split("\n")[1:])

    text = process_text(text)

    return text


def add_non_dipper_results():
    config = ExperimentConfig(
        source_datasets=['brat-project'], detectors=['DetectGPT', 'Ghostbuster'], prompt_modes=None,
        generative_models=['meta-llama/Llama-3.3-70B-Instruct', 'gpt-4o-mini-2024-07-18'],
        max_words=[None, 50, 100, 150, 200, 250],
        attack=[None]
    )
    results = load_results(config)

    for i, (info, result) in enumerate(results):
        model_name = info['model']
        if model_name == "Ghostbuster":
            model_name = "ghostbuster"
        elif model_name == "DetectGPT":
            model_name = "detect-gpt"

        if info['dataset']['info']['prompt_mode'] != "improve-human":

            llm_experiment_id = add_experiment(
                database='../../database/database.db',
                experiment_id=info['dataset']['info']['job_id'],
                dataset="argument-annotated-essays",  # rename from brat-project
                text_author=info['dataset']['info']['model'],
                prompt_mode=info['dataset']['info']['prompt_mode'],
                model=model_name,
                n_samples=info['args']['n_samples'],
                max_words=info['args']['max_words'],
                cut_sentences=info['args']['cut_sentences'],
                use_cache=False if 'use_detector_cache' not in info.keys() else info['args']['use_detector_cache'],
                human_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['human_hash'],
                llm_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['llm_hash'],
                seed=42 if 'seed' not in info.keys() else info['args']['seed'])

            df_human = get_answers("../../database/database.db", dataset="argument-annotated-essays",
                                   prompt_mode=info['dataset']['info']['prompt_mode'],
                                   generative_model=info['dataset']['info']['model'], is_human=True)

            df_llm = get_answers("../../database/database.db", dataset="argument-annotated-essays",
                                 prompt_mode=info['dataset']['info']['prompt_mode'],
                                 generative_model=info['dataset']['info']['model'], is_human=False)

            df = pd.concat([df_human, df_llm]).reset_index(drop=True)

            ds_info = info['dataset']['info']
            original_df = pd.read_csv(
                f"../datasets/{ds_info['dataset']}/{ds_info['model'].split('/')[-1].lower()}/{ds_info['job_id']}/data.csv",
                sep=";",
                index_col=0)

            def truncate_text(text):

                sentences = (nltk.sent_tokenize(text))

                processed_text = ""
                word_count = 0
                for sentence in sentences:
                    words_in_sentence = len(sentence.split())
                    if word_count + words_in_sentence >= 50:
                        if word_count == 0:
                            return np.nan
                        break
                    word_count += words_in_sentence
                    processed_text = processed_text + " " + sentence

                return processed_text

            if info['args']['max_words'] == 50:
                short_df = original_df.copy()
                short_df['human'] = short_df['human'].apply(truncate_text)
                short_df['llm'] = short_df['llm'].apply(truncate_text)
                short_df.dropna(inplace=True)

                original_df = original_df.iloc[short_df.index]

            if model_name == "detect-gpt" and info['args']['max_words'] is None:
                original_df = detect_gpt_preprocess(original_df)

            llm_df = df[df["answer"].isin(original_df['llm'].values)]

            if info['dataset']['info']['prompt_mode'] == "task":

                human_experiment_id = add_experiment(
                    database='../../database/database.db',
                    experiment_id=info['dataset']['info']['job_id'],
                    dataset="argument-annotated-essays",  # rename from brat-project
                    text_author="human",
                    prompt_mode="",
                    model=model_name,
                    n_samples=info['args']['n_samples'],
                    max_words=info['args']['max_words'],
                    cut_sentences=info['args']['cut_sentences'],
                    use_cache=False if 'use_detector_cache' not in info.keys() else info['args']['use_detector_cache'],
                    human_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['human_hash'],
                    llm_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['llm_hash'],
                    seed=42 if 'seed' not in info.keys() else info['args']['seed']
                )
                human_df = df[df["answer"].isin(original_df['human'].values)]

                predictions = result['predictions']['human']
                if len(predictions) == len(human_df) and human_experiment_id is not None:
                    add_predictions(database='../../database/database.db', experiment_id=human_experiment_id,
                                    predictions=result['predictions']['human'], answer_ids=human_df.id)

            predictions = result['predictions']['llm']

            if len(predictions) == len(llm_df) and llm_experiment_id is not None:
                add_predictions(database='../../database/database.db', experiment_id=llm_experiment_id,
                                predictions=result['predictions']['llm'], answer_ids=llm_df.id)
        else:
            # similar to the one above but human text is used
            llm_experiment_id = add_experiment(
                database='../../database/database.db',
                experiment_id=info['dataset']['info']['job_id'],
                dataset="argument-annotated-essays",  # rename from brat-project
                text_author=info['dataset']['info']['model'],
                prompt_mode=info['dataset']['info']['prompt_mode'],
                model=model_name,
                n_samples=info['args']['n_samples'],
                max_words=info['args']['max_words'],
                cut_sentences=info['args']['cut_sentences'],
                use_cache=False if 'use_detector_cache' not in info.keys() else info['args']['use_detector_cache'],
                human_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['human_hash'],
                llm_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['llm_hash'],
                seed=42 if 'seed' not in info.keys() else info['args']['seed'])

            df_human = get_answers("../../database/database.db", dataset="argument-annotated-essays",
                                   prompt_mode=info['dataset']['info']['prompt_mode'],
                                   generative_model=info['dataset']['info']['model'], is_human=True)

            df_llm = get_answers("../../database/database.db", dataset="argument-annotated-essays",
                                 prompt_mode=info['dataset']['info']['prompt_mode'],
                                 generative_model=info['dataset']['info']['model'], is_human=False)

            df = pd.concat([df_human, df_llm])

            ds_info = info['dataset']['info']
            original_df = pd.read_csv(
                f"../datasets/{ds_info['dataset']}/{ds_info['model'].split('/')[-1].lower()}/{ds_info['job_id']}/data.csv",
                sep=";",
                index_col=0)

            if model_name == "detect-gpt" and info['args']['max_words'] is None:
                original_df = detect_gpt_preprocess(original_df)

            llm_df = df[df["answer"].isin(original_df['human'].values)]

            predictions = result['predictions']['human']
            if len(predictions) == len(llm_df) and llm_experiment_id is not None:
                add_predictions(database='../../database/database.db', experiment_id=llm_experiment_id,
                                predictions=result['predictions']['human'], answer_ids=llm_df.id)


def add_dipper_results():
    config = ExperimentConfig(
        source_datasets=['brat-project'], detectors=['DetectGPT', 'Ghostbuster'], prompt_modes=None,
        generative_models=['meta-llama/Llama-3.3-70B-Instruct', 'gpt-4o-mini-2024-07-18'],
        max_words=[None],
        attack=['dipper']
    )
    results = load_results(config)

    for i, (info, result) in enumerate(results):
        model_name = info['model']
        if model_name == "Ghostbuster":
            model_name = "ghostbuster"
        elif model_name == "DetectGPT":
            model_name = "detect-gpt"

        prompt_mode = f"dipper-{info['dataset']['info']['job_id']}"

        llm_experiment_id = add_experiment(
            database='../../database/database.db',
            experiment_id=int(datetime.datetime.now().strftime('%Y%m%d%H%M%S')),
            dataset="argument-annotated-essays",  # rename from brat-project
            text_author='dipper',
            prompt_mode=prompt_mode,
            model=model_name,
            n_samples=info['args']['n_samples'],
            max_words=info['args']['max_words'],
            cut_sentences=info['args']['cut_sentences'],
            use_cache=False if 'use_detector_cache' not in info.keys() else info['args']['use_detector_cache'],
            human_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['human_hash'],
            llm_hash=None if 'input_hashes' not in info.keys() else info['input_hashes']['llm_hash'],
            seed=42 if 'seed' not in info.keys() else info['args']['seed'])

        df_human = get_answers("../../database/database.db", dataset="argument-annotated-essays",
                               prompt_mode=prompt_mode,
                               generative_model='dipper', is_human=True)

        df_llm = get_answers("../../database/database.db", dataset="argument-annotated-essays",
                             prompt_mode=prompt_mode,
                             generative_model='dipper', is_human=False)

        df = pd.concat([df_human, df_llm])

        df = df[df['is_human'] == 0]

        ds_info = info['dataset']['info']
        original_df = pd.read_csv(
            f"../datasets/{ds_info['dataset']}/dipper/{ds_info['model'].split('/')[-1].lower()}/{ds_info['job_id']}/data.csv",
            sep=";",
            index_col=0)

        if model_name == "detect-gpt" and info['args']['max_words'] is None:
            original_df = detect_gpt_preprocess(original_df)

        # When prompt_mode is "improve-human", human is actually an LLM text, but this does not matter in this case,
        # as we are not directly assign the "human" or "llm" label to prediction but only the answer_id. Since the llm text
        # is the task text, this one will not be found as get_dataset only load answers from the given prompt_mode.
        llm_df = df[df["answer"].isin(original_df['llm'].values)]

        predictions = result['predictions']['llm']
        if len(predictions) == len(llm_df) and llm_experiment_id is not None:
            add_predictions(database='../../database/database.db', experiment_id=llm_experiment_id,
                            predictions=result['predictions']['llm'], answer_ids=llm_df.id)


if __name__ == "__main__":
    add_non_dipper_results()
    add_dipper_results()