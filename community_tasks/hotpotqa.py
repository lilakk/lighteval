# Define as many as you need for your different tasks
def prompt_fn(line, task_name: str = None):
    """Defines how to go from a dataset line to a doc object.
    Follow examples in src/lighteval/tasks/default_prompts.py, or get more info
    about what this function should do in the README.
    """
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["gold"],
        instruction="",
    )

custom_metric = SampleLevelMetric(
    metric_name="my_custom_metric_name",
    higher_is_better=True,
    category=MetricCategory.IGNORED,
    use_case=MetricUseCase.NONE,
    sample_level_fn=lambda x: x,  # how to compute score for one sample
    corpus_level_fn=np.mean,  # How to aggregate the samples metrics
)

# This is how you create a simple task (like hellaswag) which has one single subset
# attached to it, and one evaluation possible.
task = LightevalTaskConfig(
    name="hotpotqa",
    prompt_function=prompt_fn,  # must be defined in the file or imported from src/lighteval/tasks/tasks_prompt_formatting.py
    suite=["community"],
    hf_repo="hotpotqa/hotpot_qa",
    hf_subset="default",
    hf_avail_splits=[],
    evaluation_splits=[],
    few_shots_split=None,
    few_shots_select=None,
    metric=[],  # select your metric in Metrics
)
