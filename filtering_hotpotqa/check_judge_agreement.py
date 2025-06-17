import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

df1 = pd.read_csv("hotpotqa_processing_results_mini.csv")
df2 = pd.read_csv("hotpotqa_processing_results.csv")

print(f"Dataset 1 shape: {df1.shape}")
print(f"Dataset 2 shape: {df2.shape}")

# merge the two dataframes on the idx column
df = pd.merge(df1, df2, on="idx", how="inner")
print(f"Merged dataset shape: {df.shape}")

# check if the correctness is the same
df["agreement"] = df["correctness_x"] == df["correctness_y"]

# print the number of agreements
print("\n" + "="*50)
print("AGREEMENT ANALYSIS")
print("="*50)
agreement_counts = df["agreement"].value_counts()
print(f"Agreement counts:")
print(agreement_counts)

# Calculate agreement rate
total_examples = len(df)
agreements = agreement_counts.get(True, 0)
disagreements = agreement_counts.get(False, 0)
agreement_rate = agreements / total_examples if total_examples > 0 else 0

print(f"\nTotal examples compared: {total_examples}")
print(f"Agreements: {agreements}")
print(f"Disagreements: {disagreements}")
print(f"Agreement rate: {agreement_rate:.4f} ({agreement_rate*100:.2f}%)")

# Calculate Cohen's Kappa for inter-rater reliability
kappa = cohen_kappa_score(df["correctness_x"], df["correctness_y"])
print(f"Cohen's Kappa: {kappa:.4f}")

# Interpretation of Kappa
if kappa < 0:
    kappa_interpretation = "Poor (worse than random)"
elif kappa < 0.20:
    kappa_interpretation = "Slight"
elif kappa < 0.40:
    kappa_interpretation = "Fair" 
elif kappa < 0.60:
    kappa_interpretation = "Moderate"
elif kappa < 0.80:
    kappa_interpretation = "Substantial"
else:
    kappa_interpretation = "Almost perfect"

print(f"Kappa interpretation: {kappa_interpretation}")

# Show distribution of correctness scores for each judge
print(f"\nJudge 1 (mini) correctness distribution:")
print(df["correctness_x"].value_counts().sort_index())
print(f"\nJudge 2 (gpt-4.1) correctness distribution:")
print(df["correctness_y"].value_counts().sort_index())

# Show examples of disagreements
disagreements_df = df[df["agreement"] == False]
if len(disagreements_df) > 0:
    print(f"\n" + "="*50)
    print("EXAMPLES OF DISAGREEMENTS")
    print("="*50)
    
    # Show first 5 disagreements
    for i, row in disagreements_df.head(5).iterrows():
        print(f"\nExample {i+1}:")
        print(f"Question: {row['question_x']}")
        print(f"Answer: {row['answer_x']}")
        print(f"Judge 1 (mini) score: {row['correctness_x']} - {row['explanation_x']}")
        print(f"Judge 2 (gpt-4.1) score: {row['correctness_y']} - {row['explanation_y']}")
        print("-" * 30)
        
    print(f"\nShowing 5 out of {len(disagreements_df)} total disagreements")
else:
    print("\nNo disagreements found!")

# Calculate agreement by question type if we can identify patterns
print(f"\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total examples: {total_examples}")
print(f"Agreement rate: {agreement_rate:.4f}")
print(f"Cohen's Kappa: {kappa:.4f} ({kappa_interpretation})")
print(f"Judge 1 accuracy: {df['correctness_x'].mean():.4f}")
print(f"Judge 2 accuracy: {df['correctness_y'].mean():.4f}")

# Save disagreements to file for further analysis
if len(disagreements_df) > 0:
    disagreements_df.to_csv("judge_disagreements.csv", index=False)
    print(f"\nDisagreements saved to 'judge_disagreements.csv'")
