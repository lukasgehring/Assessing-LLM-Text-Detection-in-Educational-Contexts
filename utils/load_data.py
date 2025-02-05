import os
import sys
from functools import partial

import nltk
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
            break
        word_count += words_in_sentence
        processed_text = processed_text + " " + sentence

    return processed_text


def load_data(args):
    df = pd.read_csv(f"datasets/{args.dataset}/data.csv", delimiter=";", index_col=0)

    if args.max_words:
        fn_truncate_text = partial(truncate_text, args=args)
        df['human'] = df['human'].apply(fn_truncate_text)
        df['llm'] = df['llm'].apply(fn_truncate_text)

        logger.info(f"Essays truncated to a maximum of {args.max_words} words.")

    logger.success(f'Loaded data: "{args.dataset}"')
    return df