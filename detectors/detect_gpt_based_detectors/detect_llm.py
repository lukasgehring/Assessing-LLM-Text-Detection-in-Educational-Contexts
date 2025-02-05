import gzip
import math
import os
import pickle
import time

import numpy as np
from tqdm import tqdm
from loguru import logger

from detectors.evaluate import run_perturbation_experiment
from detectors.utils.metrics import get_rank, get_ranks
from utils.save_data import save_results


# Modified code from: DetectLLM
# Modified code from: DetectGPT
def get_perturbation_results(results, base_model, base_tokenizer, args):
    logger.debug(f"Computing unperturbed and perturbed log rank...")
    start = time.time()

    for res in tqdm(results, desc="Computing unperturbed and perturbed log rank"):
        res["original_logrank"] = get_rank(res["original"], base_model=base_model, base_tokenizer=base_tokenizer, args=args, log=True)
        res["sampled_logrank"] = get_rank(res["sampled"], base_model=base_model, base_tokenizer=base_tokenizer, args=args, log=True)
        p_sampled_rank = get_ranks(res["perturbed_sampled"], base_model=base_model, base_tokenizer=base_tokenizer, args=args, log=True)
        p_original_rank = get_ranks(res["perturbed_original"], base_model=base_model, base_tokenizer=base_tokenizer, args=args, log=True)
        res[f"perturbed_sampled_logrank_{args.n_perturbation}"] = np.mean([i for i in p_sampled_rank[:args.n_perturbation] if not math.isnan(i)])
        res[f"perturbed_original_logrank_{args.n_perturbation}"] = np.mean([i for i in p_original_rank[:args.n_perturbation] if not math.isnan(i)])

    logger.info(f"Computed unperturbed and perturbed log rank. ({time.time() - start:.2f}s)")

    return results

def get_predictions(results, n_perturbation):
    # compute diffs with perturbed
    predictions = {'human': [], 'llm': []}
    for res in results:
        predictions['human'].append(res[f'perturbed_original_logrank_{n_perturbation}'] / res["original_logrank"])
        predictions['llm'].append(res[f'perturbed_sampled_logrank_{n_perturbation}'] / res["sampled_logrank"])

    return predictions

def run(perturbed_data, base_model, base_tokenizer, args):
    logger.info("Executing DetectLLM (NPR)")
    perturbation_results = get_perturbation_results(results=perturbed_data, base_model=base_model,
                                                               base_tokenizer=base_tokenizer,
                                                               args=args)

    predictions = get_predictions(perturbation_results, n_perturbation=args.n_perturbation)

    info = {
        'pct_words_masked': args.pct_words_masked,
        'span_length': args.span_length,
        'n_perturbations': args.n_perturbation,
        'n_samples': args.n_samples,
        'dataset': args.dataset
    }
    name = f'perturbation_{args.n_perturbation}'

    output = run_perturbation_experiment(
        results=perturbation_results,
        predictions=predictions,
        info=info,
        name=name,
        detector="DetectLLM"
    )

    save_results(
        results=output,
        model="detect-llm",
        model_name="DetectLLM",
        args=args
    )