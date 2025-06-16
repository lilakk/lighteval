import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_aggregated_data(task="community|hotpotqa|0"):
    """Load the aggregated CSV file"""
    csv_file = f"aggregated_outputs_{task.replace('|', '_')}.csv"
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"Aggregated file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    return df

def extract_f1_scores(df):
    """Extract F1 scores for all models and reshape for visualization"""
    # Get all F1 columns
    f1_columns = [col for col in df.columns if col.endswith('_f1')]
    
    # Extract model names from column names
    model_names = [col.replace('_f1', '') for col in f1_columns]
    
    # Create a long-format DataFrame for seaborn
    f1_data = []
    
    for idx, row in df.iterrows():
        for model_name, f1_col in zip(model_names, f1_columns):
            f1_score = row[f1_col]
            if pd.notna(f1_score):  # Only include non-NaN scores
                f1_data.append({
                    'example_id': idx,
                    'model': model_name,
                    'f1_score': f1_score,
                    'model_family': model_name.split('/')[0] if '/' in model_name else model_name
                })
    
    return pd.DataFrame(f1_data), model_names

def create_f1_trend_plot(df_long, model_names, save_path="f1_trends.png"):
    """Create more meaningful F1 score visualizations"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Sorted F1 scores to show performance patterns
    for model in model_names:
        model_data = df_long[df_long['model'] == model]['f1_score'].sort_values(ascending=False).reset_index(drop=True)
        short_name = model.split('/')[-1] if '/' in model else model
        ax1.plot(model_data.index, model_data.values, marker='o', markersize=3, 
                linewidth=2, alpha=0.8, label=short_name)
    
    ax1.set_title('F1 Scores Sorted by Performance (Descending)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Example Rank (Best to Worst)', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Running average F1 scores to show cumulative performance
    window_size = max(1, len(df_long[df_long['model'] == model_names[0]]) // 20)  # 5% window
    
    for model in model_names:
        model_data = df_long[df_long['model'] == model]['f1_score'].reset_index(drop=True)
        running_avg = model_data.rolling(window=window_size, center=True).mean()
        short_name = model.split('/')[-1] if '/' in model else model
        ax2.plot(running_avg.index, running_avg.values, linewidth=3, alpha=0.8, label=short_name)
    
    ax2.set_title(f'Running Average F1 Scores (Window: {window_size} examples)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Example Index', fontsize=12)
    ax2.set_ylabel('Running Average F1 Score', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()

def create_f1_distribution_plot(df_long, save_path="f1_distributions.png"):
    """Create box plots and violin plots showing F1 score distributions"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Box plot
    sns.boxplot(data=df_long, x='model', y='f1_score', ax=ax1)
    ax1.set_title('F1 Score Distributions by Model (Box Plot)', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    sns.violinplot(data=df_long, x='model', y='f1_score', ax=ax2)
    ax2.set_title('F1 Score Distributions by Model (Violin Plot)', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()

def create_heatmap(df, model_names, save_path="f1_heatmap.png"):
    """Create a heatmap showing F1 scores across examples and models"""
    # Extract F1 scores into a matrix
    f1_columns = [f'{model}_f1' for model in model_names]
    f1_matrix = df[f1_columns].values.T
    
    # Create shorter model names for better display
    short_model_names = [name.split('/')[-1] if '/' in name else name for name in model_names]
    
    # Use a reasonable fixed size that makes blocks slightly wider
    plt.figure(figsize=(30, 8))
    
    # Create heatmap with green color palette (deeper green = higher values)
    sns.heatmap(f1_matrix, 
                yticklabels=short_model_names,
                xticklabels=False,  # Too many examples to show all labels
                cmap='Greens',  # Green palette where deeper = higher values
                vmin=0, vmax=1,
                cbar_kws={'label': 'F1 Score'},
                square=False)  # Allow rectangular cells instead of forcing square
    
    plt.title('F1 Scores Heatmap: Models vs Examples', fontsize=16, fontweight='bold')
    plt.xlabel('Example Index', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    
    # Add some x-axis labels for reference
    n_examples = f1_matrix.shape[1]
    x_ticks = np.linspace(0, n_examples-1, min(10, n_examples), dtype=int)
    plt.xticks(x_ticks, x_ticks)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()

def create_model_family_comparison(df_long, save_path="family_comparison.png"):
    """Create plots comparing model families"""
    plt.figure(figsize=(14, 8))
    
    # Box plot by model family
    sns.boxplot(data=df_long, x='model_family', y='f1_score')
    plt.title('F1 Score Comparison by Model Family', fontsize=16, fontweight='bold')
    plt.xlabel('Model Family', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()

def create_summary_stats(df_long, model_names):
    """Print summary statistics"""
    print("=" * 80)
    print("F1 SCORE SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = df_long.groupby('model')['f1_score'].agg(['mean', 'std', 'min', 'max', 'count']).round(4)
    summary = summary.sort_values('mean', ascending=False)
    
    print(summary)
    print("\n")
    
    # Model family summary
    family_summary = df_long.groupby('model_family')['f1_score'].agg(['mean', 'std', 'count']).round(4)
    family_summary = family_summary.sort_values('mean', ascending=False)
    
    print("MODEL FAMILY SUMMARY:")
    print("-" * 40)
    print(family_summary)

def create_performance_ranking_plot(df_long, model_names, save_path="performance_ranking.png"):
    """Create a plot showing how models rank relative to each other on each example"""
    plt.figure(figsize=(16, 8))
    
    # Get the original dataframe structure
    pivot_df = df_long.pivot(index='example_id', columns='model', values='f1_score')
    
    # Calculate rank for each example (1 = best, higher = worse)
    rank_df = pivot_df.rank(axis=1, method='min', ascending=False)
    
    # Plot ranking trends
    for model in model_names:
        if model in rank_df.columns:
            short_name = model.split('/')[-1] if '/' in model else model
            ranks = rank_df[model].dropna()
            # Use a rolling average to smooth the ranking trends
            smoothed_ranks = ranks.rolling(window=max(1, len(ranks)//20), center=True).mean()
            plt.plot(smoothed_ranks.index, smoothed_ranks.values, 
                    linewidth=3, alpha=0.8, label=short_name, marker='o', markersize=2)
    
    plt.title('Model Performance Ranking Trends Across Examples', fontsize=16, fontweight='bold')
    plt.xlabel('Example Index', fontsize=14)
    plt.ylabel('Average Rank (1 = Best)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Lower rank numbers (better performance) at top
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all visualizations"""
    parser = argparse.ArgumentParser(description="Visualize F1 trends from aggregated model outputs.")
    parser.add_argument(
        "--task",
        type=str,
        default="community|hotpotqa|0",
        help="Task to visualize, used to find 'aggregated_outputs_{task}.csv'.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of examples to sample. If not provided, all examples are used.",
    )
    args = parser.parse_args()

    print("Loading aggregated data...")
    try:
        df = load_aggregated_data(args.task)
    except FileNotFoundError as e:
        print(e)
        return

    if args.sample_size and args.sample_size < len(df):
        print(f"Sampling {args.sample_size} examples from the dataset...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    print("Extracting F1 scores...")
    df_long, model_names = extract_f1_scores(df)

    print(f"Found {len(model_names)} models and {len(df)} examples")
    print(f"Total data points: {len(df_long)}")

    # Create a directory for charts based on the task
    output_dir = Path(f"plots_{args.task.replace('|', '_')}")
    if args.sample_size:
        output_dir = f"{output_dir}_sample_{args.sample_size}"
    output_dir.mkdir(exist_ok=True)
    print(f"Plots will be saved to '{output_dir}/'")

    # Create all visualizations
    print("\nCreating F1 trend plot...")
    create_f1_trend_plot(df_long, model_names, save_path=output_dir / "f1_trends.png")

    print("Creating F1 distribution plots...")
    create_f1_distribution_plot(df_long, save_path=output_dir / "f1_distributions.png")

    print("Creating F1 heatmap...")
    create_heatmap(df, model_names, save_path=output_dir / "f1_heatmap.png")

    print("Creating model family comparison...")
    create_model_family_comparison(df_long, save_path=output_dir / "family_comparison.png")

    print("Creating performance ranking plot...")
    create_performance_ranking_plot(
        df_long, model_names, save_path=output_dir / "performance_ranking.png"
    )

    print("Generating summary statistics...")
    create_summary_stats(df_long, model_names)

    print("\nAll visualizations completed!")

if __name__ == "__main__":
    main()
