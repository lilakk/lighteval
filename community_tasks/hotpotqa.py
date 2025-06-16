from lighteval.metrics.utils.metric_utils import SampleLevelMetric, MetricCategory, MetricUseCase
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics.metrics import Metrics
import numpy as np

# Define as many as you need for your different tasks
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
    metric=[Metrics.f1_score_hotpotqa],  # select your metric in Metrics
    generation_size=50,
    stop_sequence=["\n"]
)

TASKS_TABLE = [hotpotqa_task]
