import os
import sys

import torch
from loguru import logger

from database.interface import get_answers, get_answers_by_id
from detectors.RoBERTa.roberta_class import RoBERTa
from detectors.detect_gpt_based_detectors.detect_gpt import DetectGPT
from detectors.detect_gpt_based_detectors.fast_detect_gpt import FastDetectGPT
from detectors.ghostbuster.ghostbuster import Ghostbuster
from detectors.gptzero import GPTZeroDetector
from detectors.intrinsic_dim import IntrinsicDim
from utils.args import init_parser
from utils.logger import init_logger, log_resources
from utils.seeds import set_seeds
from utils.truncate_text import apply_max_words


def run(args):
    # remove this line, if you want to execute without a seed
    set_seeds(args.seed)

    # load full dataset or "mixed" one
    if args.dataset == 'mixed':
        with open(os.path.join("datasets", 'mixed_dataset_ids.txt'), 'r', encoding='utf-8') as f:
            ids = [int(line.rstrip('\n')) for line in f]
        df = get_answers_by_id(args.database, ids)
    else:
        df = get_answers(
            database=args.database,
            dataset=args.dataset,
            is_human=args.prompt_mode == "human",
            generative_model=args.generative_model,
            prompt_mode=args.prompt_mode
        )

    # exit if dataset is empty
    if df.empty:
        logger.error("No answers found. Please check your prompt mode, dataset and generative model parameters.")
        sys.exit(1)

    if args.max_words:
        df = apply_max_words(df, args)

    if args.model in "detect-gpt":
        logger.info("Executing DetectGPT model")
        detector = DetectGPT(args)

    elif args.model in "fast-detect-gpt":
        logger.info("Executing DetectGPT model")
        detector = FastDetectGPT(args)

    elif args.model == "intrinsic-dim":
        logger.info("Executing Intrinsic-Dim model")
        detector = IntrinsicDim(args)

    elif args.model == "ghostbuster":
        logger.info("Executing Ghostbuster model")
        detector = Ghostbuster(args)

    elif args.model == "roberta":
        logger.info("Executing RoBERTa model")
        detector = RoBERTa(args)

    elif args.model == "gpt-zero":
        logger.info("Executing GPTZero model")
        detector = GPTZeroDetector(args)

    # You can place your own detector here
    # ---------------------------------

    # ---------------------------------

    else:
        logger.error(f"Unknown model {args.model}. Please provide a valid model name!")
        sys.exit(0)

    # start detection
    detector(df)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # parse arguments
    parsed_args = init_parser()

    # init loguru logger
    parsed_args.job_id = init_logger(args=parsed_args)

    log_resources()
    logger.info(f"Run using the following arguments: {vars(parsed_args)}")

    with logger.catch():
        run(args=parsed_args)
