from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")["validation"]

print(dataset)

all_answers = [ex['answer'] for ex in dataset]

all_answers_lengths = [len(answer.split()) for answer in all_answers]

import matplotlib.pyplot as plt

plt.hist(all_answers_lengths, bins=100)
plt.savefig("hotpotqa_answer_length_distribution.png")

# check how often the new line character is used
new_line_count = sum([answer.count('\n') for answer in all_answers])
print(f"New line character count: {new_line_count}")
print(f"Percentage of new line characters: {new_line_count / len(all_answers)}")

# check how often is the answer empty
empty_answer_count = sum([answer == '' for answer in all_answers])
print(f"Empty answer count: {empty_answer_count}")
print(f"Percentage of empty answers: {empty_answer_count / len(all_answers)}")

# check how often is the answer a list
