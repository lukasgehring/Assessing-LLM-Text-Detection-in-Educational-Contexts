import numpy as np
import torch

from loguru import logger


def set_seeds(seed):
    """
    Set random seeds.
    :param seed: seed
    :return:
    """
    logger.info(f"Running on seed {seed}")

    torch.manual_seed(seed)
    np.random.seed(seed)
