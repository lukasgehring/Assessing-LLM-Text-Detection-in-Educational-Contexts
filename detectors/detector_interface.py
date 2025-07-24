import gzip
import hashlib
import os
import pickle
import time
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from loguru import logger

from database.interface import add_experiment, add_predictions, add_execution_time


class Detector(ABC):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.start_at = None

    def add_data_hash_to_args(self, human_data: pd.DataFrame, llm_data: pd.DataFrame):
        self.args.human_data_hash = None
        self.args.llm_data_hash = None

        if not human_data.empty:
            self.args.human_data_hash = hashlib.sha256(";".join(human_data.answer).encode()).hexdigest()
            logger.info(f"Human data hash: {self.args.human_data_hash}")
        if not llm_data.empty:
            self.args.llm_data_hash = hashlib.sha256(";".join(llm_data.answer).encode()).hexdigest()
            logger.info(f"LLM data hash: {self.args.llm_data_hash}")

    def __call__(self, data):
        self.start_at = time.time()
        return self.run(data)

    @abstractmethod
    def run(self, data):
        pass

    def save(self, predictions: List, answer_ids: pd.Series) -> None:
        """
        Create experiment and save predictions to database
        :param predictions: List of predictions
        :param answer_ids: IDs of the answer texts for which the predictions were made
        """

        if self.args.skip_save_to_db:
            return None

        experiment_id = add_experiment(
            database=self.args.database,
            job_id=self.args.job_id,
            dataset=self.args.dataset,
            text_author=self.args.generative_model,
            prompt_mode=self.args.prompt_mode,
            model=self.args.model,
            n_samples=self.args.n_samples,
            max_words=self.args.max_words,
            cut_sentences=self.args.cut_sentences,
            use_cache=self.args.use_detector_cache,
            human_hash=self.args.human_data_hash,
            llm_hash=self.args.llm_data_hash,
            seed=self.args.seed,
            model_checkpoint=self.args.checkpoint,
        )

        if self.start_at is not None:
            add_execution_time(
                database=self.args.database,
                experiment_id=experiment_id,
                execution_time=int(time.time() - self.start_at)
            )
        else:
            logger.warning(f"No start time found. Call the class directly (not via run()) or set self.start_time.")

        add_predictions(
            database=self.args.database,
            experiment_id=experiment_id,
            predictions=predictions,
            answer_ids=answer_ids
        )
