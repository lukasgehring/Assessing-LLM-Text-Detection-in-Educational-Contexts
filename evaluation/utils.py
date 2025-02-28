import gzip
import json
import os
import pickle
import sys
from typing import List, Tuple

import pandas as pd
from loguru import logger
from pandas import DataFrame


class ExperimentConfig:
    def __init__(self, source_datasets=None, detectors=None, prompt_modes=None, generative_models=None, max_words=None, attack=None):
        self.source_datasets = source_datasets
        self.detectors = detectors
        self.prompt_modes = prompt_modes
        self.generative_models = generative_models
        self.max_words = max_words
        self.attack = attack

def load_datasets(config: ExperimentConfig, root="../datasets"):
    info_dataset_list = []

    for path, directories, files in os.walk(root):

        info = load_info_json(path, files, config, is_dataset=True)

        if info is None:
            continue

        dataset_path = [file for file in files if '.csv' in file]
        # check if only one file exist
        assert len(dataset_path) == 1

        # load result file
        with open(os.path.join(path, dataset_path[0]), 'r') as f:
            dataset = pd.read_csv(f, sep=';', index_col=0)

        info_dataset_list.append((info, dataset))

    return info_dataset_list

def load_info_json(path, files, config:ExperimentConfig, is_dataset=False):
    if "info.json" in files:
        with open(f'{path}/info.json', 'r') as f:
            info = json.load(f)

        if not is_dataset:
            if config.detectors and info['model'] not in config.detectors:
                return None

            if info['args']['max_words'] not in config.max_words:
                return None

            dataset_info = info['dataset']['info']
        else:
            dataset_info = info['info']

        if config.generative_models and dataset_info['model'] not in config.generative_models:
            return None

        if config.source_datasets and dataset_info['dataset'] not in config.source_datasets:
            return None

        if config.prompt_modes and dataset_info['prompt_mode'] not in config.prompt_modes:
            return None

        if config.attack != info['dataset']['attack']:
            return None


        return info

def load_results(config: ExperimentConfig, root="../results", drop_keys=None):

    info_result_list = []

    for path, directories, files in os.walk(root):

        info = load_info_json(path, files, config)

        if info is None:
            continue

        result_path = [file for file in files if '.gz' in file]
        # check if only one file exist
        assert len(result_path) == 1

        # load result file
        with gzip.open(os.path.join(path, result_path[0]), 'rb') as f:
            result = pickle.load(f)

        if isinstance(result, list):
            for r in result:
                # TODO: d or z results?
                if '_d' in r['name']:
                    result: dict = r
                    break

        if drop_keys:
            for drop_key in drop_keys:
                result.pop(drop_key, None)

        info_result_list.append((info, result))

    logger.success(f"{len(info_result_list)} datasets loaded.")

    return info_result_list

def combine_info_and_metrics(info:dict, metrics:dict) -> DataFrame:
    # extract dataset info from dict
    dataset_info = info.pop('dataset')['info']

    # keep only important information and rename some keys
    dataset_info = {
        'generative_model': dataset_info['model'],
        'dataset': dataset_info['dataset'],
        'prompt_mode': dataset_info['prompt_mode'],
        'max_words': info['args']['max_words'],
    }

    # create a single row DataFrame
    return DataFrame([{**info, **dataset_info, **metrics}])