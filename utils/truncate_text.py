from argparse import Namespace
from functools import partial

import nltk
import numpy as np
import pandas as pd
from loguru import logger


def apply_max_words(df: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """
    Truncate answer columns with max words parameter.
    :param df: DataFrame with answers
    :param args: Namespace with arguments
    :return: DataFrame with truncated answers
    """

    df_cut = df.copy()

    # load nltk for sentence splitting
    nltk.download('punkt_tab')
    fn_truncate_text = partial(truncate_text, max_words=args.max_words, cut_sentences=args.cut_sentences)
    df_cut['answer'] = df_cut['answer'].apply(fn_truncate_text)

    num_rows_with_nan = df_cut.isna().any(axis=1).sum()
    df_cut.dropna(inplace=True)
    if num_rows_with_nan > 0:
        logger.debug(
            f"Dropped {num_rows_with_nan} samples due to insufficient number of words in the first sentence.")

    logger.info(f"Essays truncated to a maximum of {args.max_words} words.")

    return df_cut


def truncate_text(text: str, max_words: int, cut_sentences: bool = False) -> str:
    """
    Truncate text with max words' parameter.
    :param text: Text to truncate
    :param max_words: Maximum number of words before truncation
    :param cut_sentences: Whether to cut in between sentences
    :return: Truncated text
    """
    if cut_sentences:
        return " ".join(text.split()[:max_words])

    sentences = (nltk.sent_tokenize(text))

    processed_text = ""
    word_count = 0
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if word_count + words_in_sentence >= max_words:
            if word_count == 0:
                return np.nan
            break
        word_count += words_in_sentence
        processed_text = processed_text + " " + sentence

    return processed_text
