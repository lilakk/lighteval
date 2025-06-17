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


JUDGE_TEMPLATE = """You will be given a question, a reference answer, and a model answer. Evaluate whether the model answer is correct by assigning a score of 0 (incorrect) or 1 (correct). While the reference answer can help guide your judgment, it may not be the only valid answer, so you should also use your own knowledge to decide if the model answer is correct. Return your response in the following JSON format:

{{
    "correctness": 0 or 1
}}

Question: {question}
Reference answer: {gold}
Model answer: {pred}

Your judgement in JSON format: """


class JSONSchema(BaseModel):
    correctness: int  # 0 or 1


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


class Correctness_GPT:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        self.aggregation_function = aggregation_function
        self.model = "gpt-4.1-2025-04-14"
        self.client = Client(client_type="openai", model_name=self.model)

    @staticmethod
    def _normalize_answer(s: str) -> str:
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

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        question = kwargs['formatted_doc'].query
        results = []
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(question=question, gold=gold, pred=pred))
        return self.aggregation_function(results)
    
    def _get_judgement(self, question: str, gold: str, pred: str) -> float:
        prompt = JUDGE_TEMPLATE.format(question=question, gold=gold, pred=pred)
        response = self.client.call(
            prompt,
            model_name=self.model,
            temp=0.001,
            max_tokens=512,
            structured_output=True
        )
        return response.correctness

    def compute_one_item(self, question: str, gold: str, pred: str) -> float:
        normalized_prediction = self._normalize_answer(pred)
        normalized_gold = self._normalize_answer(gold)
        correctness = self._get_judgement(question=question, gold=normalized_gold, pred=normalized_prediction)
        return correctness


correctness_gpt = SampleLevelMetric(
    metric_name="correctness_gpt",
    sample_level_fn=Correctness_GPT().compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

extend_enum(Metrics, "correctness_gpt", correctness_gpt)


def prompt_fn(line, task_name: str = None):
    instruction = f"""Question: {line["question"]}\nConcise answer without explanation:"""
    return Doc(
        task_name=task_name,
        query=instruction,
        choices=[line["answer"]],
        gold_index=0,
    )


hotpotqa_task = LightevalTaskConfig(
    name="hotpotqa",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="yapeichang/hotpotqa-filtered",
    trust_dataset=True,
    hf_subset="",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[correctness_gpt],
    generation_size=50,
    stop_sequence=["\n"]
)

TASKS_TABLE = [hotpotqa_task]
