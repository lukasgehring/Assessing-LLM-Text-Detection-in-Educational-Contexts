import gzip
import os
import pickle
from typing import List


def load_results(name: str) -> List:
    path = os.path.join("../results", f"{name}.gz")
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        data = [data]
    return data