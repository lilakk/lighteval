import os
import glob
import argparse
import json
from datetime import datetime
import csv

models = [
    "allenai/OLMo-2-0425-1B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",

    "Qwen/Qwen2.5-3B",
    "meta-llama/Llama-3.2-3B",

    "allenai/OLMo-2-1124-7B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",

    "allenai/OLMo-2-1124-13B",
    "Qwen/Qwen2.5-14B",

    "allenai/OLMo-2-0325-32B",
    "Qwen/Qwen2.5-32B",
]


def get_file_for_model(results_dir, model, metric, task="community|hotpotqa", timestamp="latest"):
    model_dir = os.path.join(results_dir, model)
    
    if not os.path.exists(model_dir):
        return None, None
    
    # Find all JSON files in the model directory
    json_files = glob.glob(os.path.join(model_dir, "results_*.json"))
    
    if not json_files:
        print(f"No JSON files found for model {model}")
        return None, None
    
    # Filter files that contain the specified task and metric
    valid_files = []
    file_data_map = {}  # Store parsed data for each valid file
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Check if the task exists in config_tasks
            if task in data.get("config_tasks", {}):
                # Check if the metric exists in the task's metrics
                task_config = data["config_tasks"][task]
                if "metric" in task_config:
                    metrics = task_config["metric"]
                    if isinstance(metrics, list):
                        metric_names = [m.get("metric_name") for m in metrics]
                        if metric in metric_names:
                            valid_files.append(json_file)
                            file_data_map[json_file] = data
                    elif isinstance(metrics, dict) and metrics.get("metric_name") == metric:
                        valid_files.append(json_file)
                        file_data_map[json_file] = data
        except (json.JSONDecodeError, KeyError):
            # Skip files that can't be parsed or don't have the expected structure
            continue
    
    if not valid_files:
        print(f"No valid files found for model {model}, metric {metric}, task {task}")
        return None, None
    
    selected_file = None
    
    if timestamp == "latest":
        # Find the file with the latest timestamp based on filename
        # Extract timestamp from filename and parse it
        def extract_timestamp(filepath):
            filename = os.path.basename(filepath)
            # Extract timestamp part: results_YYYY-MM-DDTHH-MM-SS.microseconds.json
            try:
                timestamp_str = filename.replace("results_", "").replace(".json", "")
                # Parse the timestamp
                return datetime.fromisoformat(timestamp_str.replace("-", ":", 2))
            except:
                # Fallback to file modification time
                return datetime.fromtimestamp(os.path.getmtime(filepath))
        
        # Return the file with the latest timestamp
        selected_file = max(valid_files, key=extract_timestamp)
    else:
        # Look for a specific timestamp
        target_file = os.path.join(model_dir, f"results_{timestamp}.json")
        if target_file in valid_files:
            selected_file = target_file
        else:
            return None, None
    
    # Extract the metric score from the selected file
    if selected_file and selected_file in file_data_map:
        data = file_data_map[selected_file]
        
        # Look for the metric score in the results section
        # The task key in results might have a suffix like "|0"
        results = data.get("results", {})
        metric_score = None
        
        # Try different possible task keys in results
        for result_key in results:
            if result_key.startswith(task):
                if metric in results[result_key]:
                    metric_score = results[result_key][metric]
                    break
        
        # Also check the "all" key as a fallback
        if metric_score is None and "all" in results and metric in results["all"]:
            metric_score = results["all"][metric]
        
        return selected_file, metric_score
    
    return selected_file, None


def print_results(results_dir, model, metric, task, timestamp):
    file_path, score = get_file_for_model(results_dir, model, metric, task, timestamp)
    return model, score, file_path


def print_results_table(results_dir, models, metric, task, timestamp, csv_dir=None):
    results = []
    
    # Collect all results
    for model in models:
        model_name, score, file_path = print_results(results_dir, model, metric, task, timestamp)
        results.append((model_name, score, file_path))
    
    # Calculate column widths for alignment
    max_model_len = max(len(model) for model, _, _ in results)
    
    # Print header
    print("=" * (max_model_len + 20))
    print(f"{'Model':<{max_model_len}} | {'Score':<10} | Status")
    print("=" * (max_model_len + 20))
    
    # Print results
    for model, score, file_path in results:
        if score is not None:
            score_str = f"{score:.4f}"
            status = "✓"
        elif file_path:
            score_str = "N/A"
            status = "⚠ Found but no score"
        else:
            score_str = "N/A"
            status = "✗ Not found"
        
        print(f"{model:<{max_model_len}} | {score_str:<10} | {status}")
    
    print("=" * (max_model_len + 20))
    
    # Print summary statistics
    valid_scores = [score for _, score, _ in results if score is not None]
    if valid_scores:
        print(f"\nSummary:")
        print(f"  Models evaluated: {len(valid_scores)}/{len(results)}")
        print(f"  Best score: {max(valid_scores):.4f}")
        print(f"  Worst score: {min(valid_scores):.4f}")
    
    # Save to CSV if output path is specified
    if csv_dir:
        # Ensure the output directory exists
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        csv_output = os.path.join(csv_dir, f"{metric}_{task}.csv")
        
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Model', 'Score', 'Status', 'File_Path', 'Task', 'Metric', 'Timestamp'])
            
            # Write data rows
            for model, score, file_path in results:
                if score is not None:
                    status = "success"
                elif file_path:
                    status = "found_no_score"
                else:
                    status = "not_found"
                
                writer.writerow([
                    model,
                    score if score is not None else '',
                    status,
                    file_path if file_path else '',
                    task,
                    metric,
                    timestamp
                ])
        
        print(f"\nResults saved to: {csv_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/results")
    parser.add_argument("--metric", type=str, default="entailment_fuzzy_v2")
    parser.add_argument("--task", type=str, default="community|hotpotqa")
    parser.add_argument("--timestamp", type=str, default="latest")
    parser.add_argument("--csv_dir", type=str, default="hotpotqa/metric_results", help="Path to save CSV results file")
    args = parser.parse_args()
    
    print_results_table(args.results_dir, models, args.metric, args.task, args.timestamp, args.csv_dir)
