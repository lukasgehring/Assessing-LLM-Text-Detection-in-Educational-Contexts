import gzip
import os
import time

import nltk
import numpy as np
import dill as pickle
import tiktoken
from loguru import logger
from openai import OpenAI

from tqdm import tqdm

from detectors.evaluate import run_perturbation_experiment
from detectors.ghostbuster.utils.featurize import t_featurize_logprobs, score_ngram
from detectors.ghostbuster.utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions
from utils.save_data import save_results


def get_predictions(data, enc, best_features, model, mu, sigma,trigram_model, args):
    client = OpenAI(api_key=args.openai_key)
    predicitons = []
    MAX_TOKENS = 2047

    for text in tqdm(data):

        # Load data and featurize
        text = text.strip()
        # Strip data to first MAX_TOKENS tokens
        tokens = enc.encode(text)[:MAX_TOKENS]
        text = enc.decode(tokens).strip()


        trigram = np.array(score_ngram(text, trigram_model, enc.encode, n=3, strip_first=False))
        unigram = np.array(score_ngram(text, trigram_model.base, enc.encode, n=1, strip_first=False))

        response = client.completions.create(model="babbage-002",
        prompt="<|endoftext|>" + text,
        max_tokens=0,
        echo=True,
        logprobs=1)
        ada = np.array(list(map(lambda x: np.exp(x), response.choices[0].logprobs.token_logprobs[1:])))

        response = client.completions.create(model="davinci-002",
        prompt="<|endoftext|>" + text,
        max_tokens=0,
        echo=True,
        logprobs=1)
        davinci = np.array(list(map(lambda x: np.exp(x), response.choices[0].logprobs.token_logprobs[1:])))

        subwords = response.choices[0].logprobs.tokens[1:]
        gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
        for i in range(len(subwords)):
            for k, v in gpt2_map.items():
                subwords[i] = subwords[i].replace(k, v)

        t_features = t_featurize_logprobs(davinci, ada, subwords)

        vector_map = {
            "davinci-logprobs": davinci,
            "ada-logprobs": ada,
            "trigram-logprobs": trigram,
            "unigram-logprobs": unigram
        }

        exp_features = []
        for exp in best_features:

            exp_tokens = get_words(exp)
            curr = vector_map[exp_tokens[0]]

            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = vector_map[exp_tokens[i + 1]]
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    exp_features.append(scalar_functions[exp_tokens[i]](curr))
                    break

        # features.append(t_features + exp_features)
        data = (np.array(t_features + exp_features) - mu) / sigma
        predicitons.append(model.predict_proba(data.reshape(-1, 1).T)[:, 1])

    return np.array(predicitons).flatten()


def run(data, args):
    human = data['human'][:args.n_samples]
    llm = data['llm'][:args.n_samples]

    best_features = open("detectors/ghostbuster/model/features.txt").read().strip().split("\n")

    # Load davinci tokenizer
    enc = tiktoken.encoding_for_model("davinci-002")

    # Load model
    model = pickle.load(open("detectors/ghostbuster/model/model", "rb"))
    mu = pickle.load(open("detectors/ghostbuster/model/mu", "rb"))
    sigma = pickle.load(open("detectors/ghostbuster/model/sigma", "rb"))

    # load nltk brown dataset for trigram training
    nltk.download('brown')

    trigram_model = train_trigram()

    logger.debug(f"Compute predictions of the human-written texts.")
    start = time.time()
    human_predictions = get_predictions(
        data=human,
        enc=enc,
        best_features=best_features,
        model=model,
        mu=mu,
        sigma=sigma,
        trigram_model=trigram_model,
        args=args
    )
    logger.info(f"Finished computation of {len(human_predictions)} human-written texts. ({time.time() - start:.2f}s)")

    logger.debug(f"Compute predictions of the LLM-generated texts.")
    start = time.time()
    llm_predictions = get_predictions(
        data=llm,
        enc=enc,
        best_features=best_features,
        model=model,
        mu=mu,
        sigma=sigma,
        trigram_model=trigram_model,
        args=args
    )
    logger.info(f"Finished computation of {len(human_predictions)} LLM-generated texts. ({time.time() - start:.2f}s)")

    predictions = {'human': human_predictions.tolist(), 'llm': llm_predictions.tolist()}

    output = run_perturbation_experiment(
        results=None,
        predictions=predictions,
        name="Ghostbuster",
        info={
            'dataset': args.dataset
        },
        detector='Ghostbuster')

    save_results(
        results=output,
        model="ghostbuster",
        model_name="Ghostbuster",
        args=args
    )
