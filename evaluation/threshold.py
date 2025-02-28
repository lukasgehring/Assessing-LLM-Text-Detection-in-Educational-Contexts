import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, f1_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from evaluation.utils import load_results, ExperimentConfig


def get_threshold(config):
    info_result_list = load_results(config, drop_keys=['raw_results'])

    experiments = [result['predictions'] for _, result in info_result_list]
    infos = [info for info, _ in info_result_list]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'optimal_threshold': [],
        'accuracy': [],
        'prompt_mode': [],
        'model': []
    }

    for info, experiment in tqdm(zip(infos, experiments), total=len(infos)):
        optimal_thresholds = []
        human = np.array(experiment['human'])
        llm = np.array(experiment['llm'])

        human_train, human_test, llm_train, llm_test = train_test_split(human, llm, test_size=0.33, random_state=42)

        for train_index, val_index in kf.split(human_train):
            human_train_kf, human_val_kf = human_train[train_index], human_train[val_index]
            llm_train_kf, llm_val_kf = llm_train[train_index], llm_train[val_index]

            y_true = [0] * len(human_train_kf) + [1] * len(llm_train_kf)
            y_scores = np.concatenate((human_train_kf, llm_train_kf))

            _, _, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)

            best_f1_score = 0
            optimal_threshold = None
            for i, threshold in enumerate(thresholds):
                y_pred = [1 if score >= threshold else 0 for score in y_scores]

                current_f1_score = f1_score(y_true, y_pred)
                if current_f1_score > best_f1_score:
                    best_f1_score = current_f1_score
                    optimal_threshold = threshold

            optimal_thresholds.append(optimal_threshold)
        optimal_threshold = sum(optimal_thresholds) / len(optimal_thresholds)

        y_true = [0] * len(human_test) + [1] * len(llm_test)
        y_scores = np.concatenate((human_test, llm_test))
        y_pred = [1 if score >= optimal_threshold else 0 for score in y_scores]
        acc = accuracy_score(y_true, y_pred)

        results['optimal_threshold'].append(optimal_threshold)
        results['accuracy'].append(acc)
        results['prompt_mode'].append(info['dataset']['info']['prompt_mode'])
        results['model'].append(info['model'])


    return pd.DataFrame.from_dict(results)


if "__main__" == __name__:
    config = ExperimentConfig(
        source_datasets=['brat-project'],
        detectors=['DetectGPT'],
        generative_models=['meta-llama/Llama-3.3-70B-Instruct'],
        max_words=[None],
        # prompt_modes=['task']
    )

    print(get_threshold(config))