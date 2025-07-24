import json
import os

import requests
from loguru import logger

from detectors.detector_interface import Detector


class GPTZeroDetector(Detector):

    def __init__(self, args, **kwargs):
        super().__init__(name="GPTZero", args=args)

    def run(self, data):
        self.add_data_hash_to_args(human_data=data[data.is_human == 1], llm_data=data[data.is_human == 0])

        predictions = []
        os.makedirs(f"results/{self.args.dataset}/gpt-zero/", exist_ok=True)
        for _, text in data.iterrows():
            logger.debug(f"Request GPTZero API for answer: {text.id}")
            payload = {
                "document": text.answer,
                "multilingual": False
            }
            headers = {
                "x-api-key": "",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            response = requests.post("https://api.gptzero.me/v2/predict/text", json=payload, headers=headers)
            response_json = response.json()

            predictions.append(response_json["documents"][0]["confidence_scores_raw"]["identity"]["ai"])

            with open(f"results/{self.args.dataset}/gpt-zero/gpt-zero-response-{text.id}.json", "w",
                      encoding="utf-8") as f:
                json.dump(response_json, f, ensure_ascii=False, indent=4)

        self.save(predictions=predictions, answer_ids=data.id)
