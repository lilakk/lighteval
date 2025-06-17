import pandas as pd
import os
import glob
from datasets import load_dataset

details_dir = "results/details"
task = "community|hotpotqa|0"

models = [
    "allenai/OLMo-2-0425-1B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",

    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B",

    "allenai/OLMo-2-1124-7B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B"

    "allenai/OLMo-2-1124-13B",
    "Qwen/Qwen2.5-14B",

    "allenai/OLMo-2-0325-32B",
    "Qwen/Qwen2.5-32B",
]

all_data = {}

for model in models:
    model_family = model.split("/")[0]
    model_name = model.split("/")[1]
    path = f"{details_dir}/{model_family}/{model_name}/*/"
    timestamps = glob.glob(path)
    
    if not timestamps:
        print(f"Warning: No timestamps found for {model}")
        continue
        
    timestamp = sorted(timestamps)[-1].split("/")[-2]
    details_path = f"{details_dir}/{model_family}/{model_name}/{timestamp}/details_{task}_{timestamp}.parquet"
    
    if not os.path.exists(details_path):
        print(f"Warning: Details file not found for {model}: {details_path}")
        continue
    
    print(f"Loading data for {model}")
    details = load_dataset("parquet", data_files=details_path, split="train")
    
    for i, detail in enumerate(details):
        # Use the full prompt as the unique identifier for each row
        prompt_id = detail['full_prompt']
        
        # Initialize row if this is the first time we see this prompt
        if prompt_id not in all_data:
            all_data[prompt_id] = {
                'prompt': detail['full_prompt'],
                'gold': detail['gold'][0] if detail['gold'] else '',  # gold is a list, take first element
            }
        
        # Add this model's prediction
        prediction = detail['predictions'][0] if detail['predictions'] else ''
        f1_score = detail['metrics']['f1']
        all_data[prompt_id][f'{model}_pred'] = prediction
        all_data[prompt_id][f'{model}_f1'] = f1_score

# Convert to DataFrame
df = pd.DataFrame.from_dict(all_data, orient='index')

# Reorder columns to have prompt, gold first, then model predictions and f1 scores in models list order
base_columns = ['prompt', 'gold']
model_columns = []

# Add columns for each model in the order they appear in the models list
for model in models:
    pred_col = f'{model}_pred'
    f1_col = f'{model}_f1'
    # Only add columns that actually exist in the dataframe
    if pred_col in df.columns:
        model_columns.extend([pred_col, f1_col])

df = df[base_columns + model_columns]

output_file = f"hotpotqa/aggregated_outputs.csv"
df.to_csv(output_file, index=False)

print(f"\nSaved aggregated results to {output_file}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())