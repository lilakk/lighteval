from datasets import load_dataset
import os
import glob

output_dir = "outputs"
model_org = "allenai"
model_name = "OLMo-2-1124-7B-Instruct"
timestamp = "latest"
task = "lighteval|truthfulqa:mc|0"

if timestamp == "latest":
    path = f"{output_dir}/details/{model_org}/{model_name}/*/"
    timestamps = glob.glob(path)
    timestamp = sorted(timestamps)[-1].split("/")[-2]
    print(f"Latest timestamp: {timestamp}")

details_path = "/weka/oe-training-default/yapeic/lighteval/outputs/details/allenai/OLMo-2-1124-7B-Instruct/2025-06-13T15-07-12.143436/details_leaderboard|truthfulqa:mc|0_2025-06-13T15-07-12.143436.parquet"

details = load_dataset("parquet", data_files=details_path, split="train")

for detail in details:
    print(detail)
    import pdb; pdb.set_trace()
