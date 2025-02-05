import datetime
import os
import sys

import psutil
import torch
from loguru import logger


def log_resources():
    logger.info(
        f"System resources -> "
        f"| CPUs: {psutil.cpu_count(logical=False)} on {psutil.cpu_freq().max:.2f}MHz "
        f"| GPUs: {f'{torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})' if torch.cuda.device_count() > 0 else 0} "
        f"| RAM {psutil.virtual_memory().total / 1024 ** 3:.2f}GB "
        f"|"
    )

def init_logger(disable_log_file=False, logfile_prefix=""):
    # lof file format: 'detector-<year|month|day|hour|minute|second>[-<slurm_job_id>]'
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    timestamp = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}{f'-{slurm_job_id}' if slurm_job_id is not None else ''}"

    job_id = slurm_job_id if slurm_job_id is not None else timestamp

    log_file = f"logs/detector-{job_id}.log"

    # removing root logger and adding file logger with lvl 'INFO' and stdout logger with lvl 'DEBUG'
    logger.remove(0)
    if not disable_log_file:
        logger.add(log_file,
                   format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level> <cyan>({name}</cyan>:<cyan>{function}</cyan>:<cyan>{line})</cyan>',
                   level="INFO"
                   )
    logger.add(sys.stdout,
               format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
               )

    return job_id