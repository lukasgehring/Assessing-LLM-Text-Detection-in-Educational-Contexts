import ast
import gzip
import hashlib
import json
import os
import pickle
import sys
from functools import partial
from typing import Union, List

import nltk
import numpy as np
import pandas as pd
from loguru import logger

def truncate_text(text, args):
    if args.cut_sentences:
        return " ".join(text.split()[:args.max_words])

    sentences = (nltk.sent_tokenize(text))

    processed_text = ""
    word_count = 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if word_count + words_in_sentence >= args.max_words:
            if word_count == 0:
                return np.nan
            break
        word_count += words_in_sentence
        processed_text = processed_text + " " + sentence

    return processed_text


def load_data(args):
    df = pd.read_csv(f"datasets/{args.dataset}/data.csv", delimiter=";", index_col=0)

    if args.max_words:
        nltk.download('punkt_tab')
        fn_truncate_text = partial(truncate_text, args=args)
        df['human'] = df['human'].apply(fn_truncate_text)
        df['llm'] = df['llm'].apply(fn_truncate_text)

        num_rows_with_nan = df.isna().any(axis=1).sum()
        df.dropna(inplace=True)
        if num_rows_with_nan > 0:
            logger.debug(f"Dropped {num_rows_with_nan} samples due to insufficient number of words in the first sentence.")

        logger.info(f"Essays truncated to a maximum of {args.max_words} words.")

    args.indices = df.index

    logger.success(f'Loaded data: "{args.dataset}" | Used indices: {args.indices.values.tolist()}')
    return df

def arguments_equal(args, info, key_list):
    for arg_key, arg_value in vars(args).items():
        if arg_key in key_list:

            if arg_key == 'dataset':
                if arg_value.split("/")[0] != info['dataset']['info']['dataset']:
                    return False
                continue

            if arg_key not in info['args'].keys() or arg_value != info['args'][arg_key]:
                return False

    return True

def load_cached_data(args, model, key_list, is_human):

    data_hash = args.human_data_hash if is_human else args.llm_data_hash

    root_path = f"results"
    if os.path.exists(root_path):
        for path, directories, files in os.walk(root_path):
            if "info.json" in files:
                with open(f'{path}/info.json', 'r') as f:
                    info = json.load(f)

                if info['model'] != model:
                    continue

                if not arguments_equal(args, info, key_list=key_list):
                    continue

                if 'input_hashes' not in info.keys():
                    continue

                if (is_human and info['input_hashes']['human_hash'] == data_hash) or (info['input_hashes']['llm_hash'] == data_hash):
                    for file in files:
                        if file.endswith(".gz"):
                            with gzip.open(os.path.join(path, file), 'rb') as f:
                                result = pickle.load(f)
                            logger.info(f"Loaded cached data from {path}")
                            return result
    return None