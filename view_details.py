from datasets import load_dataset
import os
import glob

details_path = "/weka/oe-training-default/yapeic/lighteval/results/details/allenai/OLMo-2-1124-7B/2025-06-16T08-47-13.121796/details_community|hotpotqa|0_2025-06-16T08-47-13.121796.parquet"

details = load_dataset("parquet", data_files=details_path, split="train")

for detail in details:
    print(detail)
    import pdb; pdb.set_trace()
