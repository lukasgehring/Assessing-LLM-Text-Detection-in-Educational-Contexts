import gzip
import os
import pickle
import time

import numpy as np
from tqdm import tqdm

from loguru import logger

from detectors.evaluate import run_perturbation_experiment
from detectors.utils.likelihood import get_lls, get_ll
from utils.save_data import save_results


# Modified code from: DetectGPT
def get_perturbation_results(results, base_model, base_tokenizer, args):
    logger.debug(f"Computing log likelihoods...")
    start = time.time()
    for res in tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"], base_model=base_model, base_tokenizer=base_tokenizer, args=args)
        p_original_ll = get_lls(res["perturbed_original"], base_model=base_model, base_tokenizer=base_tokenizer, args=args)
        res["original_ll"] = get_ll(res["original"], base_model=base_model, base_tokenizer=base_tokenizer, args=args)
        res["sampled_ll"] = get_ll(res["sampled"], base_model=base_model, base_tokenizer=base_tokenizer, args=args)
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    logger.info(f"Computed log likelihoods. ({time.time() - start:.2f}s)")

    return results

def get_predictions(results, criterion):
    # compute diffs with perturbed
    predictions = {'human': [], 'llm': []}
    for res in results:
        if criterion == 'd':
            predictions['human'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['llm'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['human'].append(
                (res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['llm'].append(
                (res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    return predictions

def run(perturbed_data, base_model, base_tokenizer, args):
    logger.info("Executing DetectGPT")
    perturbation_results = get_perturbation_results(results=perturbed_data, base_model=base_model,
                                                               base_tokenizer=base_tokenizer, args=args)

    outputs = []
    for perturbation_mode in ['d', 'z']:
        logger.debug(f"DetectGPT inference in perturbation mode {perturbation_mode}")
        predictions = get_predictions(perturbation_results, criterion=perturbation_mode)

        info = {
            'pct_words_masked': args.pct_words_masked,
            'span_length': args.span_length,
            'n_perturbations': args.n_perturbation,
            'n_samples': args.n_samples,
            'dataset': args.dataset
        }
        name = f'perturbation_{args.n_perturbation}_{perturbation_mode}'

        output = run_perturbation_experiment(
            results=perturbation_results,
            predictions=predictions,
            info=info,
            name=name,
            detector="DetectGPT"
        )
        outputs.append(output)

    save_results(
        results=outputs,
        model="detect-gpt",
        model_name="DetectGPT",
        args=args
    )
