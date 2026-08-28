import os
import json
import pandas as pd
import numpy as np

# ================= CONFIGURATION =================
BASE_DIR = "data/final_results"

# The new w=4 files you just generated
NEW_FILES = {
    "ELI5": "ELI5_claude-3-haiku-20240307_l30_w4_eval.json",
    "WikiDialog": "WikiDialog_claude-3-haiku-20240307_l30_w4_eval.json",
    "ReWIRED": "WIRED_claude-3-haiku-20240307_l30_w4_eval.json"
}

# The w=2 files to compare against (Standard Naming Assumption)
# If your files are named differently, edit these!
BASELINE_FILES = {
    "ELI5": "ELI5_claude-3-haiku-20240307_l30_w2_eval_tuned_eval.json",
    "WikiDialog": "WikiDialog_claude-3-haiku-20240307_l30_w2_eval_tuned_eval.json",
    "ReWIRED": "WIRED_claude-3-haiku-20240307_l30_w2_eval_tuned_eval.json"
}

# FED Metrics list
METRICS = [
    "engaging", "specific", "relevant", "correct", "semantically appropriate",
    "understandable", "fluent", "coherent", "error recovery", "consistent",
    "diverse", "depth", "likeable", "understand", "flexible", "informative", "inquisitive"
]


# ================= HELPERS =================

def load_scores(filepath):
    """Robustly loads FED scores from JSON."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Robust Load logic (same as before)
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        if "model_output" in data or "task_output" in data:
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame.from_dict(data, orient='index')

    # Filter for just numeric FED columns
    available_metrics = [m for m in METRICS if m in df.columns]
    return df[available_metrics].mean()


# ================= MAIN =================

def main():
    print(f"{'DATASET':<12} | {'METRIC':<20} | {'w=2':<8} | {'w=4':<8} | {'DIFF':<8}")
    print("-" * 70)

    results_summary = {}

    for dataset, new_file in NEW_FILES.items():
        # Paths
        path_w4 = os.path.join(BASE_DIR, dataset, new_file)

        # Handle folder naming variations (e.g. WIRED vs ReWIRED)
        # If 'ReWIRED' folder doesn't exist, try 'WIRED'
        if not os.path.exists(os.path.dirname(path_w4)):
            alt_path = path_w4.replace("ReWIRED", "WIRED")
            if os.path.exists(os.path.dirname(alt_path)):
                path_w4 = alt_path

        path_w2 = os.path.join(os.path.dirname(path_w4), BASELINE_FILES[dataset])

        # Load Data
        scores_w4 = load_scores(path_w4)
        scores_w2 = load_scores(path_w2)

        if scores_w4 is None:
            print(f"Skipping {dataset}: w=4 file not found.")
            continue

        if scores_w2 is None:
            print(f"Skipping {dataset}: Baseline w=2 file not found ({BASELINE_FILES[dataset]}).")
            # Just print w4 if w2 is missing
            print(f"\n=== {dataset} (w=4 Only) ===")
            print(scores_w4.to_string())
            continue

        # Compare
        diffs = scores_w4 - scores_w2

        # Store average absolute change to summarize later
        avg_change = diffs.abs().mean()
        results_summary[dataset] = avg_change

        print(f"\n=== {dataset} COMPARISON ===")
        for metric in METRICS:
            if metric in scores_w4 and metric in scores_w2:
                v2 = scores_w2[metric]
                v4 = scores_w4[metric]
                d = v4 - v2

                # Highlight big changes
                marker = " (!)" if abs(d) > 1.0 else ""

                print(f"{dataset:<12} | {metric:<20} | {v2:.2f}     | {v4:.2f}     | {d:+.2f}{marker}")

    print("\n\n=== SUMMARY FOR REBUTTAL ===")
    for ds, change in results_summary.items():
        print(f"{ds}: Average score change when doubling context window: {change:.3f}")


if __name__ == "__main__":
    main()
