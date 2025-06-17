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


JUDGE_TEMPLATE = """You are an expert at evaluating the correctness of answers to a question. You will be provided with a question and two answers to the question. For each answer, determine its correctness by assigning a score of 0 or 1 (0 means incorrect, 1 means correct), and provide a short explanation for your judgement. Return your response in the following JSON format:

{{
    "judgement": {{
        "answer1": {{
            "correctness": 0 or 1,
            "explanation": short explanation for your judgement
        }}
        "answer2": {{
            "correctness": 0 or 1,
            "explanation": short explanation for your judgement
        }}
    }}
}}

Question: {question}
Answer 1: {answer1}
Answer 2: {answer2}

Your judgement in JSON format: """


class AnswerJudgement(BaseModel):
    correctness: int  # 0 or 1
    explanation: str

class Judgement(BaseModel):
    answer1: AnswerJudgement
    answer2: AnswerJudgement

class JSONSchema(BaseModel):
    judgement: Judgement


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

    def call(self, prompt, model_name=None, temp=0.001, max_tokens=512):
        if self.client_type == "openai":
            return self.openai_call(prompt, model_name, temp, max_tokens)
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")


class Correctness_GPT:
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        self.aggregation_function = aggregation_function
        self.model = "gpt-4.1-mini"
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
        prompt = JUDGE_TEMPLATE.format(question=question, answer1=gold, answer2=pred)
        response = self.client.call(
            prompt,
            model_name=self.model,
            temp=0.001,
            max_tokens=512,
            structured_output=True
        )
        return response.judgment.answer2.correctness

    def compute_one_item(self, question: str, gold: str, pred: str) -> float:
        """
        Use LLM-based evaluation.
        """
        normalized_prediction = self._normalize_answer(pred)
        normalized_gold = self._normalize_answer(gold)
        judgement = self._get_judgement(question=question, gold=normalized_gold, pred=normalized_prediction)
        pass


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
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/default_prompts.py, or get more info
    about what this function should do in the README.
    """
    
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
    hf_repo="hotpotqa/hotpot_qa",
    trust_dataset=True,
    hf_subset="fullwiki",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    metric=[correctness_gpt],
    generation_size=50,
    stop_sequence=["\n"]
)

TASKS_TABLE = [hotpotqa_task]
