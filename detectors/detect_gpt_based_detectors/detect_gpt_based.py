import functools
import sys
import time

import torch
import transformers

from loguru import logger

from detectors.utils.load_hf import hf_load_pretrained_llm
from detectors.utils.perturb import perturb_texts
from detectors.utils.preprocess_data import preprocess_data
from detectors.detect_gpt_based_detectors import detect_gpt, detect_llm

from utils.args import add_data_hash_to_args
from utils.load_data import load_cached_data


def perturb_data(data, mask_model, mask_tokenizer, args):
    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    # result object contains every input text and their perturbed samples
    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx]
        })

    perturb_fn = functools.partial(perturb_texts, span_length=args.span_length, pct=args.pct_words_masked,
                                   mask_model=mask_model, mask_tokenizer=mask_tokenizer, args=args)

    if args.use_detector_cache:
        # searching for perturbations from previous results
        hash_key_list = ['buffer_size', 'cut_sentences', 'dataset', 'mask_filling_model_name', 'mask_top_p',
                         'max_num_attempts', 'max_words', 'n_perturbation', 'pct_words_masked', 'span_length', 'seed',
                         'n_samples']

        # TODO: What to do with DetectLLM?
        cached_human_data = load_cached_data(args, model="DetectGPT", key_list=hash_key_list, is_human=True)
        cached_llm_data = load_cached_data(args, model="DetectGPT", key_list=hash_key_list, is_human=False)
    else:
        cached_human_data = None
        cached_llm_data = None

    if cached_human_data is not None and cached_llm_data is not None:
        logger.warning(
            "Experiment stopped! Experiment with the same configuration was done before. Delete the old experiment or change arguments!")
        sys.exit(0)

    human_loaded, llm_loaded = False, False

    # loading human data if exist
    if cached_human_data:
        for idx, result in enumerate(cached_human_data[0]['raw_results']):
            for k in ['original', 'sampled', 'perturbed_sampled', 'sampled_ll', 'all_perturbed_sampled_ll',
                      'perturbed_sampled_ll', 'perturbed_sampled_ll_std']:
                result.pop(k, None)

            for key, value in result.items():
                results[idx][key] = value
        logger.info("Loaded cached human data.")
        human_loaded = True

    # loading llm data if exist
    elif cached_llm_data:
        for idx, result in enumerate(cached_llm_data[0]['raw_results']):
            for k in ['sampled', 'original', 'perturbed_original', 'original_ll', 'all_perturbed_original_ll',
                      'perturbed_original_ll', 'perturbed_original_ll_std']:
                result.pop(k, None)

            for key, value in result.items():
                results[idx][key] = value
        logger.info("Loaded cached LLM data.")
        llm_loaded = True

    # generating human data
    if not human_loaded:
        logger.debug(f"Generate perturbations of the human-written texts.")
        start = time.time()
        p_original_text = perturb_fn([x for x in original_text for _ in range(args.n_perturbation)])
        logger.info(
            f"Finished generation of {len(p_original_text)} perturbations of the human-written texts. ({time.time() - start:.2f}s)")

        assert len(p_original_text) == len(
            original_text) * args.n_perturbation, f"Expected {len(original_text) * args.n_perturbation} perturbed samples, got {len(p_original_text)}"

        for idx, res in enumerate(results):
            res["perturbed_original"] = p_original_text[idx * args.n_perturbation: (idx + 1) * args.n_perturbation]

    # generating llm data
    if not llm_loaded:
        logger.debug(f"Generate perturbations of the LLM-generated texts.")
        start = time.time()
        p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(args.n_perturbation)])
        logger.info(
            f"Finished generation of {len(p_sampled_text)} perturbations of the LLM-generated texts. ({time.time() - start:.2f}s)")

        assert len(p_sampled_text) == len(
            sampled_text) * args.n_perturbation, f"Expected {len(sampled_text) * args.n_perturbation} perturbed samples, got {len(p_sampled_text)}"

        for idx, res in enumerate(results):
            res["perturbed_sampled"] = p_sampled_text[idx * args.n_perturbation: (idx + 1) * args.n_perturbation]

    # unload mask model to save vram
    logger.debug("Unload mask model")
    del mask_model
    del mask_tokenizer
    torch.cuda.empty_cache()

    return results


def run(models, data, args):
    # loading huggingface pretrained models. Add correct model_class to load the correct model!
    mask_model, mask_tokenizer = hf_load_pretrained_llm(args.mask_filling_model_name,
                                                        model_class=transformers.AutoModelForSeq2SeqLM,
                                                        cache_dir=args.cache_dir)

    # strip whitespaces, newlines and keep only samples with <= 512 token
    data = preprocess_data(data, mask_tokenizer=mask_tokenizer, args=args)

    # add_data_hash_to_args(args=args, human_data=data['original'], llm_data=data['sampled'])

    # perturb human-written and llm-generated texts
    perturbed_data = perturb_data(data, mask_model, mask_tokenizer, args)

    base_model, base_tokenizer = hf_load_pretrained_llm(args.base_model_name, cache_dir=args.cache_dir)

    if "detect-gpt" in models:
        detect_gpt.run(perturbed_data, base_model, base_tokenizer, args)
    if "detect-llm" in models:
        detect_llm.run(perturbed_data, base_model, base_tokenizer, args)
