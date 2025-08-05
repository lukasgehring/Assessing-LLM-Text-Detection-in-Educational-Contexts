import datetime
import os
import sys

from loguru import logger


def init_logger(args, logfile_prefix=""):
    # lof file format: 'detector-<year|month|day|hour|minute|second>[-<slurm_job_id>]'
    slurm_job_id = os.getenv('SLURM_JOB_ID')
    timestamp = f"{datetime.datetime.now().strftime('%y%m%d%H%M%S')}{f'-{slurm_job_id}' if slurm_job_id is not None else ''}"

    job_id = slurm_job_id if slurm_job_id is not None else timestamp

    log_file = f"logs/experiment-{job_id}.log"

    # removing root logger and adding file logger with lvl 'INFO' and stdout logger with lvl 'DEBUG'
    logger.remove(0)
    if not args.disable_log_file:
        logger.add(log_file,
                   format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level> <cyan>({name}</cyan>:<cyan>{function}</cyan>:<cyan>{line})</cyan>',
                   level="INFO"
                   )
    logger.add(sys.stdout,
               format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>'
               )

    return job_id
