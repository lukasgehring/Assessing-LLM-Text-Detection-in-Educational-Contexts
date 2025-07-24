import argparse
import hashlib
import os
import sys
from datetime import datetime
from typing import List

import pandas as pd
import torch
from loguru import logger


def init_parser():
    parser = argparse.ArgumentParser(prog='BenchEduLLMDetect')
    parser.add_argument('--model', type=str, default="detect-gpt", help="detector-models")

    parser.add_argument('--dataset', default="argument-annotated-essays", help="dataset path")
    parser.add_argument('--database', default="../database/database.db", help="database path")
    parser.add_argument('--prompt_mode', default="task",
                        help="available modes: task (default), summary, task+summary, rewrite-human, improve-human, rewrite-[job_id], dipper-[job_id], human")
    parser.add_argument('--generative_model', default="meta-llama/Llama-3.3-70B-Instruct",
                        help="dataset generative model")

    parser.add_argument('--device', default="cuda", help="dataset")
    parser.add_argument('--n_samples', default=5, type=int, help="Number of samples")
    parser.add_argument('--chunk_size', default=20, type=int, help="Chunk size")  # TODO: Maybe combine with batch_size?
    parser.add_argument('--base_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3b")
    parser.add_argument('--cache_dir', type=str, default=".resources")
    parser.add_argument('--max_num_attempts', type=int, default=20, help="Number of rewriting attempts")
    parser.add_argument('--disable_log_file', action='store_true')
    parser.add_argument('--skip_save_to_db', action='store_true')
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

    if "roberta" == args.model:
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

    if args.prompt_mode[:7] == "dipper-":
        args.generative_model = "dipper"
    elif args.prompt_mode == "human":
        args.generative_model = "human"

    return args
