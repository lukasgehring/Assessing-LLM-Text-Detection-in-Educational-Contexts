import functools
import os
import pickle
import time

import transformers

from loguru import logger

from detectors.utils.load_hf import model_to_device, hf_load_pretrained_llm
from detectors.utils.perturb import perturb_texts
from detectors.utils.preprocess_data import preprocess_data
from detectors.detect_gpt_based_detectors import detect_gpt, detect_llm


def perturb_data(data, mask_model, mask_tokenizer, args):
    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, span_length=args.span_length, pct=args.pct_words_masked,
                                   mask_model=mask_model, mask_tokenizer=mask_tokenizer, args=args)

    # generate perturbations of llm-texts
    logger.debug(f"Generate perturbations of the LLM-generated texts.")
    start = time.time()
    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(args.n_perturbation)])
    logger.info(f"Finished generation of {len(p_sampled_text)} perturbations of the LLM-generated texts. ({time.time() - start:.2f}s)")

    # generate perturbations of human-texts
    logger.debug(f"Generate perturbations of the human-written texts.")
    start = time.time()
    p_original_text = perturb_fn([x for x in original_text for _ in range(args.n_perturbation)])
    logger.info(f"Finished generation of {len(p_sampled_text)} perturbations of the human-written texts. ({time.time() - start:.2f}s)")

    assert len(p_sampled_text) == len(
        sampled_text) * args.n_perturbation, f"Expected {len(sampled_text) * args.n_perturbation} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(
        original_text) * args.n_perturbation, f"Expected {len(original_text) * args.n_perturbation} perturbed samples, got {len(p_original_text)}"

    # result object contains every input text and their perturbed samples
    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * args.n_perturbation: (idx + 1) * args.n_perturbation],
            "perturbed_original": p_original_text[idx * args.n_perturbation: (idx + 1) * args.n_perturbation]
        })

    # unload mask model to save vram
    logger.debug("Unload mask model")
    del mask_model
    del mask_tokenizer

    return results


def run(models, data, args):
    # loading huggingface pretrained models. Add correct model_class to load the correct model!
    mask_model, mask_tokenizer = hf_load_pretrained_llm(args.mask_filling_model_name, model_class=transformers.AutoModelForSeq2SeqLM, cache_dir=args.cache_dir)

    # strip whitespaces, newlines and keep only samples with <= 512 token
    data = preprocess_data(data, mask_tokenizer=mask_tokenizer, args=args)

    # perturb human-written and llm-generated texts
    perturbed_data = perturb_data(data, mask_model, mask_tokenizer, args)

    base_model, base_tokenizer = hf_load_pretrained_llm(args.base_model_name, cache_dir=args.cache_dir)

    if "detect-gpt" in models:
        detect_gpt.run(perturbed_data, base_model, base_tokenizer, args)
    if "detect-llm" in models:
        detect_llm.run(perturbed_data, base_model, base_tokenizer, args)


