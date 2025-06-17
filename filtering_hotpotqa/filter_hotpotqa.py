from lighteval.metrics.utils.metric_utils import SampleLevelMetric, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
import numpy as np

import re
import string
from typing import Callable
from nltk.metrics.distance import edit_distance
from collections import Counter
from aenum import extend_enum
import os
from openai import OpenAI
from pydantic import BaseModel
from datasets import load_dataset
from tqdm import tqdm
import concurrent.futures
import threading
import time
import pandas as pd

# Thread-safe lock for shared operations
lock = threading.Lock()

dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")["validation"]
# dataset = dataset.select(range(500))
print(f"Loaded dataset with {len(dataset)} examples")

JUDGE_TEMPLATE = """You will be provided with a question and an answer. Determine the correctness of the answer by assigning a score of 0 or 1 (0 means incorrect, 1 means correct). If you decide to assign a score of 0, also provide a short explanation for your judgement, otherwise leave it empty. Return your response in the following JSON format:

{{
    "correctness": 0 or 1,
    "explanation": short explanation for your judgement (can be empty if correctness is 1)
}}

Question: {question}
Answer: {answer}

Your judgement in JSON format: """


class JSONSchema(BaseModel):
    correctness: int  # 0 or 1
    explanation: str


class Client():
    def __init__(self, client_type, model_name=None):
        self.client_type = client_type
        self.model_name = model_name
        if client_type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Invalid client type: {client_type}")
    
    def openai_call(self, prompt, model_name=None, temp=0.001, max_tokens=512, structured_output=True):
        messages = [
            {"role": "user", "content": prompt}
        ]
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "seed": 42,
            "max_completion_tokens": max_tokens,
            "temperature": temp
        }
        if structured_output:
            completion_kwargs["response_format"] = JSONSchema
        
        if not structured_output:
            response = self.client.chat.completions.create(**completion_kwargs)
            return response.choices[0].message.content.strip()
        else:
            response = self.client.beta.chat.completions.parse(**completion_kwargs)
            return response.choices[0].message.parsed

    def call(self, prompt, model_name=None, temp=0.001, max_tokens=512, structured_output=True):
        if self.client_type == "openai":
            return self.openai_call(prompt, model_name, temp, max_tokens, structured_output)
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")


class Validator:
    def __init__(
        self,
    ):
        # self.model = "gpt-4.1-mini"
        self.model = "gpt-4.1"
        self.client = Client(client_type="openai", model_name=self.model)

    @staticmethod
    def _normalize(s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _get_judgement(self, question: str, answer: str) -> tuple:
        prompt = JUDGE_TEMPLATE.format(question=question, answer=answer)
        response = self.client.call(
            prompt,
            model_name=self.model,
            temp=0.001,
            max_tokens=512,
            structured_output=True
        )
        correctness = response.correctness
        explanation = response.explanation
        return correctness, explanation

    def check_answer(self, question: str, answer: str) -> tuple:
        """
        Use LLM-based evaluation.
        """
        normalised_question = self._normalize(question)
        normalised_answer = self._normalize(answer)
        correctness, explanation = self._get_judgement(question=normalised_question, answer=normalised_answer)
        return correctness, explanation


def process_example(example, idx, validator):
    """Process a single example with retry logic"""
    max_retries = 5
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            correctness, explanation = validator.check_answer(example["question"], example["answer"])
            
            with lock:
                process_example.num_processed += 1
                
            return {
                "idx": idx,
                "question": example["question"],
                "answer": example["answer"],
                "correctness": correctness,
                "explanation": explanation
            }
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Error processing example {idx} after {max_retries} retries: {e}")
                return {
                    "idx": idx,
                    "question": example["question"],
                    "answer": example["answer"],
                    "correctness": 0,
                    "explanation": f"Error during evaluation: {str(e)}"
                }
            else:
                time.sleep(5)  # Wait before retry

# Initialize counter as a static variable
process_example.num_processed = 0


def process_dataset_multithreaded(dataset, max_workers=20, save_steps=100):
    """Process the dataset using multithreading"""
    print(f"Processing {len(dataset)} examples with {max_workers} workers...")
    
    # Initialize validator (one per thread will be created)
    validator = Validator()
    
    # Check if we have existing results
    save_path = "hotpotqa_processing_results.csv"
    all_results = []
    processed_indices = set()
    
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        all_results = existing_df.to_dict(orient='records')
        processed_indices = set(existing_df['idx'].tolist())
        print(f"Loaded {len(all_results)} existing results from {save_path}")
    
    # Create list of examples that need processing
    examples_to_process = []
    for idx, example in enumerate(dataset):
        if idx not in processed_indices:
            examples_to_process.append((example, idx))
    
    print(f"Need to process {len(examples_to_process)} new examples")
    
    # Reset counter
    process_example.num_processed = 0
    
    # Process examples using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a separate validator for each worker to avoid shared state issues
        futures = []
        for example, idx in examples_to_process:
            # Each thread gets its own validator instance
            thread_validator = Validator()
            future = executor.submit(process_example, example, idx, thread_validator)
            futures.append(future)
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating answers"):
            result = future.result()
            if result:
                all_results.append(result)
                
                # Save intermediate results
                if len(all_results) % save_steps == 0:
                    with lock:
                        df = pd.DataFrame(all_results)
                        df.to_csv(save_path, index=False)
                        print(f"Saved {len(all_results)} results to {save_path}")
    
    # Save final results
    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)
    print(f"Final save: {len(all_results)} results to {save_path}")
    
    return all_results


# Process all examples with multithreading
results = process_dataset_multithreaded(dataset, max_workers=20, save_steps=100)

# Convert results back to the original format for dataset modification
correctness_scores = [0] * len(dataset)
explanations = [""] * len(dataset)

for result in results:
    idx = result["idx"]
    correctness_scores[idx] = result["correctness"]
    explanations[idx] = result["explanation"]

# Add new columns to the dataset
dataset_with_scores = dataset.add_column("correctness", correctness_scores)
dataset_with_scores = dataset_with_scores.add_column("explanation", explanations)

# Save the modified dataset
print("Saving modified dataset...")
dataset_with_scores.save_to_disk("hotpotqa_with_correctness")

# Print aggregate statistics
total_examples = len(correctness_scores)
correct_answers = sum(correctness_scores)
incorrect_answers = total_examples - correct_answers
accuracy = correct_answers / total_examples if total_examples > 0 else 0

print("\n" + "="*50)
print("AGGREGATE STATISTICS")
print("="*50)
print(f"Total examples processed: {total_examples}")
print(f"Correct answers: {correct_answers}")
print(f"Incorrect answers: {incorrect_answers}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("="*50)

# Print some examples of incorrect answers for inspection
print("\nExamples of incorrect answers:")
print("-" * 30)
incorrect_count = 0
for i, (ex, correctness, explanation) in enumerate(zip(dataset, correctness_scores, explanations)):
    if correctness == 0:
        incorrect_count += 1
        print(f"\nExample {i+1}:")
        print(f"Question: {ex['question']}")
        print(f"Answer: {ex['answer']}")
        print(f"Explanation: {explanation}")
        print("-" * 30)
        if incorrect_count >= 5:  # Show only first 5 incorrect examples
            break

print(f"\nDataset saved to 'hotpotqa_with_correctness' directory")
print(f"Intermediate results saved to 'hotpotqa_processing_results.csv'")
