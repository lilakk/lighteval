from datasets import load_from_disk

dataset = load_from_disk("hotpotqa_with_correctness")
print(dataset)
#Dataset({
#    features: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context', 'correctness', 'explanation'],
#    num_rows: 7405
#})

# create a new dataset where the correctness is 1
dataset_correct = dataset.filter(lambda x: x["correctness"] == 1)

# save the dataset
dataset_correct.save_to_disk("hotpotqa_filtered")

# upload the dataset to the hub
dataset_correct.push_to_hub("yapeichang/hotpotqa-filtered")
