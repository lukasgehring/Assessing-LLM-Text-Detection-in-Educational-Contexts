import argparse
import hashlib
import os
import sys
from datetime import datetime
from typing import List

import torch
from loguru import logger


def add_data_hash_to_args(args: argparse.Namespace, human_data: List[str], llm_data: List [str]):
    args.human_data_hash = hashlib.sha256(";".join(human_data).encode()).hexdigest()
    args.llm_data_hash = hashlib.sha256(";".join(llm_data).encode()).hexdigest()

    logger.info(f"Human data hash: {args.human_data_hash} | LLM data hash: {args.llm_data_hash}")

def init_parser():
    parser = argparse.ArgumentParser(prog='BenchEduLLMDetect')
    parser.add_argument('--models', nargs='+', default=["detect-gpt", "detect-llm", "intrinsic-dim", "ghostbuster"], help="detector-models")
    parser.add_argument('--dataset', default="brat-project/llama-3.3-70b-instruct/131538", help="dataset path")
    parser.add_argument('--device', default="cuda", help="dataset")
    parser.add_argument('--n_samples', default=5, type=int, help="Number of samples")
    parser.add_argument('--chunk_size', default=20, type=int, help="Chunk size") #  TODO: Maybe combine with batch_size?
    parser.add_argument('--base_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3b")
    parser.add_argument('--cache_dir', type=str, default=".resources")
    parser.add_argument('--max_num_attempts', type=int, default=20, help="Number of rewriting attempts")
    parser.add_argument('--disable_log_file', action='store_true')
    parser.add_argument("--openai_key", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    # dataset restrictions
    parser.add_argument("--max_words", type=int, default=None)
    parser.add_argument("--cut_sentences", action='store_true')

    # detector cache
    parser.add_argument('--use_detector_cache', action='store_true')

    # roberta
    parser.add_argument('--checkpoint', default=None, type=str, help="Checkpoint path for pretrained RoBERTa model")


    # TODO: Maybe fix the following arguments?
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--n_perturbation', type=int, default=100)
    parser.add_argument('--span_length', type=int, default=2)

    args = parser.parse_args()

    if "roberta" in args.models:
        if args.checkpoint is None:
            sys.exit("Please provide a checkpoint path for RoBERTa model!")
        if not os.path.exists(args.checkpoint):
            sys.exit(f"RoBERTa model checkpoint path {args.checkpoint} does not exist!")

    args.start_at = datetime.now()

    if args.device != "cpu":
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    return args