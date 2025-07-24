import sys
from abc import ABC

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from detectors.detector_interface import Detector


class RoBERTa(Detector, ABC):
    def __init__(self, args, **kwargs):
        super().__init__(name="RoBERTa", args=args)
        self.pipe = self.load_pretrained_roberta()

    @staticmethod
    def prepare_dataset(data):
        df_human = pd.DataFrame({"text": data[data.is_human == 1].answer, "label": 0})
        df_llm = pd.DataFrame({"text": data[data.is_human == 0].answer, "label": 1})
        return Dataset.from_pandas(pd.concat([df_human, df_llm], ignore_index=True)), len(df_human), len(df_llm)

    def load_pretrained_roberta(self):
        return pipeline('text-classification', model=self.args.checkpoint, device_map=self.args.device, framework='pt')

    def run(self, data):
        self.add_data_hash_to_args(human_data=data[data.is_human == 1], llm_data=data[data.is_human == 0])
        dataset, num_human_samples, num_llm_samples = self.prepare_dataset(data)

        predictions = []
        for out in tqdm(self.pipe(KeyDataset(dataset, "text"), batch_size=8, max_length=512, truncation=True),
                        total=len(dataset)):
            if out['label'] == "LABEL_0":
                predictions.append(1 - out['score'])
            else:
                predictions.append(out['score'])

        self.save(predictions, answer_ids=data.id)

        return predictions, data.id
