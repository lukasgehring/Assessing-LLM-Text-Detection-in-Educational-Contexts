import time

import nltk
import numpy as np
import dill as pickle
import tiktoken
from loguru import logger
from openai import OpenAI

from tqdm import tqdm

from detectors.detector_interface import Detector
from detectors.ghostbuster.utils.featurize import t_featurize_logprobs, score_ngram
from detectors.ghostbuster.utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions


class Ghostbuster(Detector):

    def __init__(self, args):
        super().__init__(name='Ghostbuster', args=args)

        self.enc = None
        self.best_features = None
        self.model = None
        self.mu = None
        self.sigma = None
        self.trigram_model = None

        self.best_features = open("detectors/ghostbuster/model/features.txt").read().strip().split("\n")

        # Load davinci tokenizer
        self.enc = tiktoken.encoding_for_model("davinci-002")

        # Load model
        self.model = pickle.load(open("detectors/ghostbuster/model/model", "rb"))
        self.mu = pickle.load(open("detectors/ghostbuster/model/mu", "rb"))
        self.sigma = pickle.load(open("detectors/ghostbuster/model/sigma", "rb"))

        # load nltk brown dataset for trigram training
        nltk.download('brown')

        self.trigram_model = train_trigram()

    def get_predictions(self, data):
        client = OpenAI(api_key=self.args.openai_key)
        predicitons = []
        MAX_TOKENS = 2047

        for text in tqdm(data.answer):

            # Load data and featurize
            text = text.strip()
            # Strip data to first MAX_TOKENS tokens
            tokens = self.enc.encode(text)[:MAX_TOKENS]
            text = self.enc.decode(tokens).strip()

            trigram = np.array(score_ngram(text, self.trigram_model, self.enc.encode, n=3, strip_first=False))
            unigram = np.array(score_ngram(text, self.trigram_model.base, self.enc.encode, n=1, strip_first=False))

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
            for exp in self.best_features:

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
            features = (np.array(t_features + exp_features) - self.mu) / self.sigma
            predicitons.append(self.model.predict_proba(features.reshape(-1, 1).T)[:, 1])

        return np.array(predicitons).flatten().tolist()

    def run(self, data):
        data = data.head(self.args.n_samples)

        self.add_data_hash_to_args(human_data=data[data.is_human == 1], llm_data=data[data.is_human == 0])

        logger.debug(f"Compute predictions of the texts.")
        start = time.time()
        predictions = self.get_predictions(
            data=data
        )
        logger.info(f"Finished computation of {len(predictions)} human-written texts. ({time.time() - start:.2f}s)")

        self.save(predictions=predictions, answer_ids=data.id)

        return predictions, data.id
