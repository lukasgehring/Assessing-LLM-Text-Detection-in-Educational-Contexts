import re
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, f1_score, accuracy_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, KFold

from database.interface import get_predictions


def map_dipper_to_generative_model(row):
    if row.generative_model == "dipper":
        match = re.search(r'\d+', row.prompt_mode)
        if match:
            number = int(match.group())
            row.generative_model = 'gpt-4o-mini-2024-07-18' if number >= 20000000 else 'meta-llama/Llama-3.3-70B-Instruct'
    return row


def filter_and_get_roc_auc(sub_df, prompt_mode):
    """
    Filter DataFrame based on prompt_mode and return ROC-AUC score.
    """
    if prompt_mode == "human":
        return None

    if sub_df[sub_df['prompt_mode'] == prompt_mode].is_human.all():
        filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "task")]
    else:
        filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "human")]

    return get_roc_auc(filtered_df)

def filter_and_get_tpr_at_fpr(sub_df, prompt_mode, target_fpr=0.05):
    """
    Filter DataFrame based on prompt_mode and return ROC-AUC score.
    """
    if prompt_mode == "human":
        return None

    if sub_df[sub_df['prompt_mode'] == prompt_mode].is_human.all():
        filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "task")]
    else:
        filtered_df = sub_df[(sub_df['prompt_mode'] == prompt_mode) | (sub_df['prompt_mode'] == "human")]

    return get_tpr_at_fpr(filtered_df, target_fpr)


def get_roc_auc(df):
    fpr, tpr, _ = roc_curve(~df['is_human'], df['prediction'])
    score = auc(fpr, tpr)
    # switch label if score is below 0.5
    if score < 0.5:
        print("WARNING: Flipping true label in auc")
        fpr, tpr, _ = roc_curve(df['is_human'], df['prediction'])
        return auc(fpr, tpr)
    return score

def get_tpr_at_fpr(df, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(~df['is_human'], df['prediction'])


    if auc(fpr, tpr) < 0.5:
        print("WARNING: Flipping true label in tpr")
        fpr, tpr, _ = roc_curve(df['is_human'], df['prediction'])

    # Interpolation: TPR bei exakt 0.05 FPR
    return np.interp(target_fpr, fpr, tpr)


def get_pr_auc(df):
    fpr, tpr, _ = roc_curve(~df['is_human'], df['prediction'])

    precision, recall, _ = precision_recall_curve(~df['is_human'], df['prediction'])

    score = auc(fpr, tpr)
    # switch label if score is below 0.5
    if score < 0.5:
        print("WARNING: Flipping true label in pr_auc")
        precision, recall, _ = precision_recall_curve(df['is_human'], df['prediction'])

    return auc(recall, precision)


def set_label(df, boundary="rewrite-human"):
    PROMPT_ORDER = ["improve-human", "rewrite-human", "summary", "task+summary"]
    boundary_index = PROMPT_ORDER.index(boundary)
    prior_prompts = set(PROMPT_ORDER[:boundary_index + 1])
    mask = df['prompt_mode'].isin(prior_prompts)
    df.loc[mask, 'is_human'] = True


def get_data(generative_model: str, dataset: str = None, detector: str = None,
             database: str = "../database/database.db") -> pd.DataFrame:
    # load generative_model predictions
    df = get_predictions(
        database=database,
        dataset=dataset,
        prompt_mode=None,
        generative_model=generative_model,
        detector=detector,
        is_human=None,
    )

    df.loc[df['is_human'] == True, 'prompt_mode'] = 'human'

    # load dipper predictions
    dipper = get_predictions(
        database=database,
        dataset=dataset,
        prompt_mode=None,
        generative_model="dipper",
        detector=detector,
        is_human=False,
    )

    dipper = dipper[dipper['prompt_mode'] != '']

    # make sure, to load the correct dipper results
    dipper['number'] = dipper['prompt_mode'].str.extract(r'dipper-(\d+)').astype(int)
    if "gpt" in generative_model:
        dipper = dipper[dipper['number'] >= 20000000]
    else:
        dipper = dipper[dipper['number'] <= 20000000]
    dipper = dipper.drop(columns=['number'])

    # combine predictions
    return pd.concat([df, dipper]).reset_index(drop=True)


def get_threshold(df, invert=False):
    _, _, thresholds = roc_curve(y_true=~df.is_human.to_numpy(), y_score=df.prediction.to_numpy(),
                                 drop_intermediate=False)

    best_f1_score = 0
    optimal_threshold = None
    for i, threshold in enumerate(thresholds):
        y_pred = np.array([1 if score >= threshold else 0 for score in df.prediction.to_numpy()])
        if invert:
            y_pred = 1 - y_pred

        current_f1_score = f1_score(~df.is_human.to_numpy(), y_pred)
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            optimal_threshold = threshold

    return optimal_threshold


def get_threshold_kFold(df, invert=False):
    X_train, X_test, y_train, y_test = train_test_split(df.prediction.to_numpy(),
                                                        ~df.is_human.to_numpy(),
                                                        test_size=0.33,
                                                        random_state=42)

    optimal_thresholds = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train):
        X_train_kf, X_val_kf = X_train[train_index], X_train[val_index]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

        _, _, thresholds = roc_curve(y_true=y_train_kf, y_score=X_train_kf, drop_intermediate=False)

        best_f1_score = 0
        optimal_threshold = None
        for i, threshold in enumerate(thresholds):
            y_pred = np.array([1 if score >= threshold else 0 for score in X_train_kf])
            if invert:
                y_pred = 1 - y_pred

            current_f1_score = f1_score(y_train_kf, y_pred)
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                optimal_threshold = threshold

        optimal_thresholds.append(optimal_threshold)

    return sum(optimal_thresholds) / len(optimal_thresholds)


def get_accuracy(df):
    threshold = get_threshold(df)

    df["y_pred"] = (df["prediction"] >= threshold).astype(int)
    accuracy = accuracy_score(~df["is_human"], df["y_pred"])
    print(accuracy)


def remove_rows_by_condition(df: pd.DataFrame, conditions: dict):
    """
    Remove rows of a DataFrame based on conditions from a dict.

    :param df: DataFrame
    :param conditions: Dictionary with {Column: Value or List of Values}. Rows that fulfill conditions are removed.
    :return: DataFrame with rows removed.
    """
    for column, values in conditions.items():
        if isinstance(values, list):
            df = df[~df[column].isin(values)]
        else:
            df = df[df[column] != values]
    return df


def select_best_roberta_checkpoint(df, dataset_column_name="name"):
    not_roberta = df['detector'] != 'roberta'

    roberta = (
            ((df[dataset_column_name] == 'argument-annotated-essays') &
             (df['model_checkpoint'] == 'detectors/RoBERTa/checkpoints/persuade/binary/checkpoint-24'))
            |
            ((df[dataset_column_name] != 'argument-annotated-essays') &
             (df[
                  'model_checkpoint'] == 'detectors/RoBERTa/checkpoints/argument-annotated-essays/binary/checkpoint-62'))
    )
    mask = not_roberta | roberta

    return df[mask].reset_index(drop=True)


def compute_mean_std(df, columns, score_column_name, drop_mean=True, drop_std=True):
    df_grouped = df.groupby(columns)[score_column_name].agg(['mean', 'std']).reset_index()
    df_grouped['ROC-AUC (Mean ± Std)'] = df_grouped.apply(
        lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1
    )
    if drop_mean:
        df_grouped.drop(columns=["mean"], inplace=True)
    if drop_std:
        df_grouped.drop(columns=["std"], inplace=True)

    return df_grouped


def remove_job_id_from_prompt_mode(df):
    df['prompt_mode'] = df['prompt_mode'].replace(r"^rewrite-\d+$", "rewrite-llm", regex=True)
    df['prompt_mode'] = df['prompt_mode'].replace(r"^dipper-\d+$", "dipper", regex=True)


# highlight best scores
def highlight_max(df, best_row=True, best_col=True, add_sign=False):
    df_fmt = df.copy().astype(str)

    row_max = df.max(axis=1)

    for col in df.columns:
        col_max = df[col].max()
        for i in df.index:
            val = df.at[i, col]

            if add_sign and val >= 0:
                cell_str = f"+{val:.2f}"
            else:
                cell_str = f"{val:.2f}"

            if val == row_max[i] and best_row:
                cell_str = f"\\underline{{{cell_str}}}"

            if val == col_max and best_col:
                cell_str = f"\\textbf{{{cell_str}}}"

            df_fmt.at[i, col] = cell_str

    return df_fmt


if __name__ == "__main__":
    pass
