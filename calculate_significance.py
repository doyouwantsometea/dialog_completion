import json
import os
import pandas as pd
from scipy import stats


# ================= CONFIGURATION =================
BASE_DIR = "data/final_results"
DATASET = "ELI5"

FILE_MODEL_A = f"{DATASET}_Meta-Llama-3.1-8B-Instruct_l30_w2_eval_tuned_eval.json"
FILE_MODEL_B = f"{DATASET}_Mistral-7B-Instruct-v0.3_l30_w2_eval_tuned_eval.json"

# Standard FED Metrics
METRICS = [
    "engaging", "specific", "relevant", "correct", "semantically appropriate",
    "understandable", "fluent", "coherent", "error recovery", "consistent",
    "diverse", "depth", "likeable", "understand", "flexible", "informative", "inquisitive"
]


# ================= HELPERS =================

def load_file_robust(filepath):
    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    # Robust Load
    if isinstance(raw_data, list):
        df = pd.DataFrame(raw_data)
    elif isinstance(raw_data, dict):
        # Check if column-oriented
        if "model_output" in raw_data or "task_output" in raw_data or "index" in raw_data:
            df = pd.DataFrame(raw_data)
        else:
            # ID-oriented
            df = pd.DataFrame.from_dict(raw_data, orient='index')

    # We combine 'file' and 'index' to avoid merging Turn 1 of Dialogue A with Turn 1 of Dialogue B
    if 'file' in df.columns and 'index' in df.columns:
        df['merge_key'] = df['file'].astype(str) + "_" + df['index'].astype(str)
    else:
        # Fallback if columns missing
        print("Warning: 'file' or 'index' column missing. Merging might be inaccurate.")
        df['merge_key'] = df.index.astype(str)

    return df


def calculate_cohens_d(x, y):
    diff = x - y
    # Avoid division by zero
    if diff.std() == 0: return 0.0
    return diff.mean() / diff.std()


# ================= ANALYSIS =================

def run_comparisons():
    path_a = os.path.join(BASE_DIR, DATASET, FILE_MODEL_A)
    path_b = os.path.join(BASE_DIR, DATASET, FILE_MODEL_B)

    print(f"Loading A: {FILE_MODEL_A}")
    df_a = load_file_robust(path_a)
    print(f"Loading B: {FILE_MODEL_B}")
    df_b = load_file_robust(path_b)

    # 1. Model A vs Model B
    merged = pd.merge(df_a, df_b, on='merge_key', suffixes=('_A', '_B'), how='inner')

    print(f"\nAligned {len(merged)} unique turns for comparison.")
    print(f"{'METRIC':<25} | {'WINNER':<10} | {'P-VALUE':<10} | {'EFFECT (d)':<10} | {'SIG'}")
    print("-" * 85)

    for metric in METRICS:
        col_a = f"{metric}_A"  # Because of suffix
        col_b = f"{metric}_B"

        if col_a not in merged.columns:
            continue

        scores_a = merged[col_a]
        scores_b = merged[col_b]

        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        d_val = calculate_cohens_d(scores_a, scores_b)

        mean_diff = scores_a.mean() - scores_b.mean()
        winner = "Llama" if mean_diff > 0 else "Mistral"
        is_sig = p_val < 0.05
        sig_mark = "**" if is_sig else ""

        print(f"{metric:<25} | {winner:<10} | {p_val:.2e}   | {abs(d_val):.2f}       | {sig_mark}")

    # 2. Self-Refinement Check
    print(f"\n\n=== SELF-REFINEMENT CHECK ({FILE_MODEL_A}) ===")
    print(f"{'METRIC':<25} | {'CHANGE':<10} | {'P-VALUE':<10} | {'EFFECT (d)':<10} | {'SIG'}")
    print("-" * 85)

    for metric in METRICS:
        col_task = metric
        col_tuned = f"{metric}-tuned"

        if col_task not in df_a.columns or col_tuned not in df_a.columns:
            # Skip if metric missing (e.g. synonym density sometimes excluded)
            continue

        # Create a temporary clean dataframe for this specific metric pair
        clean_df = df_a[[col_task, col_tuned]].dropna()

        if len(clean_df) < 2:
            print(f"{metric:<25} | SKIPPED (N<2)")
            continue

        scores_task = clean_df[col_task]
        scores_tuned = clean_df[col_tuned]

        # Paired T-Test
        t_stat, p_val = stats.ttest_rel(scores_tuned, scores_task)
        d_val = calculate_cohens_d(scores_tuned, scores_task)

        # Determine direction
        mean_diff = scores_tuned.mean() - scores_task.mean()
        direction = "IMPROVED" if mean_diff > 0 else "DECLINED"

        # Mark significance
        sig_mark = "**" if p_val < 0.05 else ""

        print(f"{metric:<25} | {direction:<10} | {p_val:.2e}   | {d_val:.2f}       | {sig_mark}")


if __name__ == "__main__":
    run_comparisons()
