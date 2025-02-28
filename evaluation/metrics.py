import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import roc_curve, confusion_matrix, auc
from tqdm import tqdm

from evaluation.utils import combine_info_and_metrics


def get_accuracy(true_positives: ndarray, true_negatives: ndarray, false_positives: ndarray, false_negatives: ndarray) -> ndarray:
    return (true_negatives + true_positives) / (true_positives + true_negatives + false_positives + false_negatives)


def get_precision(true_positives: ndarray, false_positives: ndarray, _epsilon: float = 1e-7) -> ndarray:
    return true_positives / (true_positives + false_positives + _epsilon)


def get_recall(true_positives: ndarray, false_negatives: ndarray, _epsilon: float = 1e-7) -> ndarray:
    return true_positives / (true_positives + false_negatives + _epsilon)


def get_f1_score(precision: ndarray, recall: ndarray, _epsilon: float = 1e-7) -> ndarray:
    return 2 * (precision * recall) / (precision + recall + _epsilon)


def compute_metrics(info_result_list: List[Tuple[dict, dict]]) -> Optional[DataFrame]:
    df = None

    for info, result in tqdm(info_result_list, desc="Computing metrics"):
        # setup y_true (0 = human, 1 = LLM) and y_scores [2 * num_samples]
        y_true = [0] * len(result['predictions']['human']) + [1] * len(result['predictions']['llm'])
        y_scores = result['predictions']['human'] + result['predictions']['llm']

        # computing fpr and tpr for every threshold
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)

        num_thresholds = len(thresholds)

        true_positives = np.zeros(num_thresholds)
        false_negatives = np.zeros(num_thresholds)
        false_positives = np.zeros(num_thresholds)
        true_negatives = np.zeros(num_thresholds)

        # computing confusion matrix for each possible threshold
        for i, threshold in enumerate(thresholds):
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_pred).ravel()

            true_positives[i] = true_positive
            false_negatives[i] = false_negative
            false_positives[i] = false_positive
            true_negatives[i] = true_negative

        # computing additional metrics
        precision = get_precision(true_positives=true_positives, false_positives=false_positives)
        recall = get_recall(true_positives=true_positives, false_negatives=false_negatives)

        # TODO: Add human and llm recall
        metrics = {
            'false_positive_rate': false_positive_rate,
            'true_positive_rate': true_positive_rate,
            'true_positive': true_positives,
            'false_negative': false_negatives,
            'false_positive': false_positives,
            'true_negative': true_negatives,
            'roc_auc': auc(false_positive_rate, true_positive_rate),
            'accuracy': get_accuracy(true_positives=true_positives, true_negatives=true_negatives, false_positives=false_positives, false_negatives=false_negatives),
            'precision': precision,
            'recall': recall,
            'f1_score': get_f1_score(precision=precision, recall=recall)
        }

        df = pd.concat([df, combine_info_and_metrics(info, metrics)], ignore_index=True) if df is not None else combine_info_and_metrics(info, metrics)

    logger.info("Finished computing metrics.")

    return df