import time

import torch
from loguru import logger

from detectors import intrinsic_dim
from detectors.detect_gpt_based_detectors import detect_gpt_based
from detectors.ghostbuster import ghostbuster
from utils.args import init_parser
from utils.load_data import load_data
from utils.logger import init_logger, log_resources
from utils.seeds import set_seeds


def run(args):

    # remove this line, if you want to execute without a seed
    set_seeds(42)

    # loading data (questions, human-written and llm-generated texts)
    data = load_data(args)

    # iterate through all defined detector models and group them (if possible)
    detect_gpt_based_models = []
    for model in args.models:

        # all detect-gpt based models
        if model in ["detect-gpt", "detect-llm"]:
            detect_gpt_based_models.append(model)

        if model == "intrinsic-dim":
            logger.info("Executing Intrinsic-Dim model")
            args.start_timestamp = time.time()
            intrinsic_dim.run(data, args)

        if model == "ghostbuster":
            logger.info("Executing Ghostbuster model")
            args.start_timestamp = time.time()
            ghostbuster.run(data, args)

    # execute detectors (inference and evaluation)
    if len(detect_gpt_based_models) > 0:
        start = time.time()
        logger.info("Executing DetectGPT-based model(s)")
        args.start_timestamp = time.time()
        detect_gpt_based.run(detect_gpt_based_models, data, args)
        logger.info(f"DetectGPT-based model(s) finished! ({time.time() - start: .2f}s)")



if __name__ == "__main__":
    torch.cuda.empty_cache()

    # parse arguments
    args = init_parser()

    # init loguru logger
    args.job_id = init_logger(disable_log_file=args.disable_log_file)

    log_resources()
    logger.info(f"Run using the following arguments: {vars(args)}")

    with logger.catch():
        run(args=args)

