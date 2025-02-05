import numpy as np
from nltk import ngrams


def get_token_len(tokens):
    """
    Returns a vector of word lengths, in tokens
    """
    tokens_len = []
    curr = 0

    for token in tokens:
        if token[0] == "Ġ":
            tokens_len.append(curr)
            curr = 1
        else:
            curr += 1

    return np.array(tokens_len)




def score_ngram(doc, model, tokenizer, n=3, strip_first=False):
    """
    Returns vector of ngram probabilities given document, model and tokenizer
    """
    scores = []
    if strip_first:
        doc = " ".join(doc.split()[:1000])
    for i in ngrams((n - 1) * [50256] + tokenizer(doc.strip()), n):
        scores.append(model.n_gram_probability(i))

    return np.array(scores)


def t_featurize_logprobs(davinci_logprobs, ada_logprobs, tokens):
    X = []

    outliers = []
    for logprob in davinci_logprobs:
        if logprob > 3:
            outliers.append(logprob)

    X.append(len(outliers))
    outliers += [0] * (50 - len(outliers))
    X.append(np.mean(outliers[:25]))
    X.append(np.mean(outliers[25:50]))

    diffs = sorted(davinci_logprobs - ada_logprobs, reverse=True)
    diffs += [0] * (50 - min(50, len(diffs)))
    X.append(np.mean(diffs[:25]))
    X.append(np.mean(diffs[25:]))

    token_len = sorted(get_token_len(tokens), reverse=True)
    token_len += [0] * (50 - min(50, len(token_len)))
    X.append(np.mean(token_len[:25]))
    X.append(np.mean(token_len[25:]))

    return X