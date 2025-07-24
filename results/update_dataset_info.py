import json
import os

for path, directories, files in os.walk("."):
    if "info.json" in files:
        with open(os.path.join(path, "info.json")) as f:
            info = json.load(f)

        dataset_info_path = f"../datasets/brat-project/{info['dataset']['info']['model'].split('/')[-1].lower()}/{info['dataset']['info']['job_id']}/info.json"
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)


        info['dataset'] = dataset_info

        with open(os.path.join(path, "info.json"), 'w') as f:
            json.dump(info, f, indent=4)