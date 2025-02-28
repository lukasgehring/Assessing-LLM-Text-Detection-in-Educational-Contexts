import gzip
import json
import os
import pickle
import time

from loguru import logger


def create_info(args, path, model):
    arg_dict = vars(args).copy()
    infos = {'model': model}

    infos['input_hashes'] = {
        'human_hash': arg_dict.pop("human_data_hash"),
        'llm_hash': arg_dict.pop("llm_data_hash"),
    }

    if "openai_key" in arg_dict.keys():
        arg_dict.pop("openai_key")

    if "indices" in arg_dict.keys():
        arg_dict.pop("indices")

    if "start_at" in arg_dict.keys():
        start_at = arg_dict.pop("start_at")
        infos['start_at'] = str(start_at)

    if "start_timestamp" in arg_dict.keys():
        start_timestamp = arg_dict.pop("start_timestamp")
        infos['duration'] = f"{time.time() - start_timestamp:.2f}s"

    if "dataset" in arg_dict.keys():
        dataset_path = arg_dict.pop("dataset")
        try:
            with open(f"datasets/{dataset_path}/info.json", "r") as f:
                infos['dataset'] = json.load(f)
        except FileNotFoundError:
            pass

    infos['args'] = arg_dict



    with open(f"{path}/info.json", 'w') as f:
        json.dump(infos, f, indent=4)

def save_results(results, model, model_name, args):
    folder = f"results/{args.dataset}/{model}/{args.job_id}"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{model}_{args.job_id}.gz")
    with gzip.open(path, "wb") as f:
        pickle.dump(results, f)

    create_info(args, path=folder, model=model_name)
    logger.success(f"{model_name} results saved at {path}")