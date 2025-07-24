import sys

import pandas as pd
import sklearn
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from detectors.evaluate import run_perturbation_experiment
from utils.args import add_data_hash_to_args
from utils.save_data import save_results


def prepare_dataset(data):
    df_human = pd.DataFrame({"text": data["human"], "label": 0})
    df_llm = pd.DataFrame({"text": data["llm"], "label": 1})
    return Dataset.from_pandas(pd.concat([df_human, df_llm], ignore_index=True))

def load_pretrained_roberta(args):
    return pipeline('text-classification', model=args.checkpoint, device_map=args.device, framework='pt')

def run(data, args):
    add_data_hash_to_args(args=args, human_data=data['human'].tolist(), llm_data=data['llm'].tolist())
    dataset = prepare_dataset(data)

    pipe = pipeline('text-classification', model=args.checkpoint, device="cuda", framework='pt')

    predictions = []
    for out in tqdm(pipe(KeyDataset(dataset, "text"), batch_size=8, max_length=512, truncation=True), total=len(dataset)):
        if out['label'] == "LABEL_0":
            predictions.append(1 - out['score'])
        else:
            predictions.append(out['score'])


    predictions = {'human': predictions[:int(len(dataset)/2)], 'llm': predictions[int(len(dataset)/2):]}

    output = run_perturbation_experiment(
        results=None,
        predictions=predictions,
        name="RoBERTa",
        info={
            'dataset': args.dataset
        },
        detector='RoBERTa')

    save_results(
        results=output,
        model="roberta",
        model_name="RoBERTa",
        args=args
    )

if __name__ == "__main__":
    df = pd.read_csv("../../datasets/brat-project/llama-3.3-70b-instruct/131538/data.csv", sep=";", index_col=0)
    df_mix = pd.read_csv("../../datasets/brat-project/llama-3.3-70b-instruct/135465/data.csv", sep=";", index_col=0)

    pipe = pipeline('text-classification',
                    model="checkpoints/ielts/llama-3.3-70b-instruct/135672/checkpoint-21",
                    device="cuda",
                    framework='pt',
                    #function_to_apply="none",
                    return_all_scores=True)

    samples = 100

    y_true = [0] * samples + [1] * samples + [2] * samples
    y_pred = []

    for data in tqdm(df['human'][:samples]):
        print(pipe(data, max_length=512, truncation=True))
        #sys.exit(0)
        #y_pred.append(int(pipe(data, max_length=512, truncation=True)[0]['label'][-1]))

    for data in tqdm(df['llm'][:samples]):
        print(pipe(data, max_length=512, truncation=True))
        #y_pred.append(int(pipe(data, max_length=512, truncation=True)[0]['label'][-1]))

    #for data in tqdm(df_mix['human'][:samples]):
    #    y_pred.append(int(pipe(data, max_length=512, truncation=True)[0]['label'][-1]))


    #print(sklearn.metrics.accuracy_score(y_true, y_pred))