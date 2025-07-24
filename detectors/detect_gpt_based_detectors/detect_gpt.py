import functools
import time
from typing import List

import torch
import transformers
import re
import numpy as np

from tqdm import tqdm
from loguru import logger

from detectors.detector_interface import Detector
from detectors.utils.likelihood import get_lls, get_ll
from detectors.utils.load_hf import hf_load_pretrained_llm


# Modified code from: DetectGPT
def get_perturbation_results(results, base_model, base_tokenizer, args):
    logger.debug(f"Computing log likelihoods...")
    start = time.time()
    for res in tqdm(results, desc="Computing log likelihoods"):

        # if data not cached, compute human results
        if "original_ll" not in res.keys():
            p_original_ll = get_lls(res["perturbed_original"], base_model=base_model, base_tokenizer=base_tokenizer,
                                    args=args)
            res["original_ll"] = get_ll(res["original"], base_model=base_model, base_tokenizer=base_tokenizer,
                                        args=args)
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    logger.info(f"Computed log likelihoods. ({time.time() - start:.2f}s)")

    return results


# Modified code from: DetectGPT
def tokenize_and_mask(text, span_length, pct, args):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    n_spans = int(n_spans)

    # Added to prevent infinite regeneration loop
    # TODO: Model have problems when using more than 27 spans! See this issue: https://github.com/huggingface/transformers/issues/8842
    if n_spans > 27:
        n_spans = 27

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


# Modified code from: DetectGPT
def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# Modified code from: DetectGPT
# replace each masked span with a sample from T5 mask_model
def replace_masks(mask_tokenizer, mask_model, texts, args):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p,
                                  num_return_sequences=1, eos_token_id=stop_id)

    # set stop token if model execute with </s>
    # ids = (outputs == stop_id).any(dim=1).flatten()
    # if not ids.all():
    #    idx = (outputs[(~ids).nonzero()] == 1).nonzero()[0]
    #    outputs[(~ids).nonzero(),idx] = stop_id
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


# Modified code from: DetectGPT
def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    # define regex to match all <extra_id_*> tokens, where * is an integer
    pattern = re.compile(r"<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


# Modified code from: DetectGPT
def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


# Modified code from: DetectGPT
def perturb_texts_(texts, span_length, pct, mask_tokenizer, mask_model, args):
    masked_texts = [tokenize_and_mask(x, span_length, pct, args) for x in texts]
    raw_fills = replace_masks(mask_tokenizer, mask_model, masked_texts, args)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        logger.warning(f'{len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, args) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(mask_tokenizer, mask_model, masked_texts, args)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

        # Add reduce inference time (following DetectLLM)
        if attempts > args.max_num_attempts:
            break

    return perturbed_texts


# Modified code from: DetectGPT
def perturb_texts(texts, span_length, pct, mask_model, mask_tokenizer, args):
    chunk_size = args.chunk_size

    outputs = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, mask_tokenizer, mask_model, args))
    return outputs


def strip_newlines(text):
    return ' '.join(text.split())


def preprocess_data(data, mask_tokenizer, args):
    # remove whitespaces
    logger.debug("Strip whitespaces and newlines from human-written and llm-generated texts...")

    data.answer = data.answer.apply(lambda x: x.strip())
    data.answer = data.answer.apply(strip_newlines)

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    logger.debug("Remove samples with more than 512 tokens.")

    tokenized_answers = mask_tokenizer(data.answer.to_list())

    mask = [len(x) <= 512 for x in tokenized_answers["input_ids"]]
    data = data[mask].head(args.n_samples)

    logger.info(
        f"Prepared data with {len(data)} texts, with a maximum of 512 tokens according to mask_tokenizer.")

    return data


def perturb_data(data, mask_model, mask_tokenizer, args):
    results = []

    perturb_fn = functools.partial(perturb_texts, span_length=args.span_length, pct=args.pct_words_masked,
                                   mask_model=mask_model, mask_tokenizer=mask_tokenizer, args=args)

    # generate perturbations of llm-texts
    logger.debug(f"Generate perturbations of texts.")
    start = time.time()
    p_answer = perturb_fn([x for x in data.answer for _ in range(args.n_perturbation)])
    logger.info(
        f"Finished generation of {len(p_answer)} perturbations of the LLM-generated texts. ({time.time() - start:.2f}s)")

    assert len(p_answer) == len(
        data.answer) * args.n_perturbation, f"Expected {len(data.answer) * args.n_perturbation} perturbed samples, got {len(p_answer)}"

    # result object contains every input text and their perturbed samples
    for i, idx in enumerate(data.index):
        results.append({
            "original": data.answer[idx],
            "perturbed_original": p_answer[i * args.n_perturbation: (i + 1) * args.n_perturbation]
        })

    # unload mask model to save vram
    logger.debug("Unload mask model")

    mask_model.to("cpu")
    torch.cuda.empty_cache()

    del mask_model
    del mask_tokenizer

    # make sure, the model is deleted
    time.sleep(1)

    return results


class DetectGPT(Detector):
    def __init__(self, args):
        super().__init__("detect-gpt", args)

    def get_predictions(self, data) -> List:
        predictions = []
        for res in data:
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                logger.warning("std of perturbed original is 0, setting to 1")
                logger.warning(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                logger.warning(f"Original text: {res['original']}")

            predictions.append(
                (res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])

        return predictions

    def run(self, data):
        # loading huggingface pretrained models. Add correct model_class to load the correct model!
        mask_model, mask_tokenizer = hf_load_pretrained_llm(self.args.mask_filling_model_name,
                                                            model_class=transformers.AutoModelForSeq2SeqLM,
                                                            cache_dir=self.args.cache_dir)

        # strip whitespaces, newlines and keep only samples with <= 512 token
        data = preprocess_data(data, mask_tokenizer=mask_tokenizer, args=self.args)

        self.add_data_hash_to_args(human_data=data[data.is_human == 1], llm_data=data[data.is_human == 0])

        # perturb human-written and llm-generated texts
        perturbed_data = perturb_data(data, mask_model, mask_tokenizer, self.args)

        base_model, base_tokenizer = hf_load_pretrained_llm(self.args.base_model_name, cache_dir=self.args.cache_dir)

        logger.info("Executing DetectGPT")
        perturbation_results = get_perturbation_results(results=perturbed_data, base_model=base_model,
                                                        base_tokenizer=base_tokenizer, args=self.args)

        predictions = self.get_predictions(data=perturbation_results)

        self.save(predictions=predictions, answer_ids=data.id)

        return predictions, data.id
