import numpy as np
from nltk.corpus import brown
import tiktoken

from detectors.ghostbuster.utils.n_gram import TrigramBackoff


vec_functions = {
    "v-add": lambda a, b: a + b,
    "v-sub": lambda a, b: a - b,
    "v-mul": lambda a, b: a * b,
    "v-div": lambda a, b: np.divide(
        a, b, out=np.zeros_like(a), where=(b != 0), casting="unsafe"
    ),
    "v->": lambda a, b: a > b,
    "v-<": lambda a, b: a < b,
}

scalar_functions = {
    "s-max": max,
    "s-min": min,
    "s-avg": lambda x: sum(x) / len(x),
    "s-avg-top-25": lambda x: sum(sorted(x, reverse=True)[:25])
    / len(sorted(x, reverse=True)[:25]),
    "s-len": len,
    "s-var": np.var,
    "s-l2": np.linalg.norm,
}


def get_words(exp):
    """
    Splits up expression into words, to be individually processed
    """
    return exp.split(" ")


def train_trigram(verbose=True, return_tokenizer=False):
    """
    Trains and returns a trigram model on the brown corpus
    """

    enc = tiktoken.encoding_for_model("davinci-002")
    tokenizer = enc.encode
    vocab_size = enc.n_vocab

    # We use the brown corpus to train the n-gram model
    sentences = brown.sents()

    tokenized_corpus = []
    for sentence in sentences:
        tokens = tokenizer(" ".join(sentence))
        tokenized_corpus += tokens

    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus)
