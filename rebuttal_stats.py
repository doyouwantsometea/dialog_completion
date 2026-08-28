"""Per-dataset paired-significance table for the ARR rebuttal.

For each (dataset, model) cell on the vanilla prompt, run a paired t-test
between the task output and the iteratively self-refined output on every
FED and IXQuisite metric. Counts how many of M metrics show significant
improvement (tuned > task at p<0.05) and how many show significant
degradation. Echoes the headline ELI5 / Llama numbers already in the
paper and extends the same check to ReWIRED and WikiDialog so the
rebuttal can claim per-dataset robustness.
"""
import os
import pandas as pd
from scipy import stats

BASE_DIR = "data/final_results"

DATASETS = ["WIRED", "WikiDialog", "ELI5"]
DISPLAY = {"WIRED": "ReWIRED", "WikiDialog": "WikiDialog", "ELI5": "ELI5"}
MODELS = {
    "Llama-3.1": "Meta-Llama-3.1-8B-Instruct",
    "Mistral-0.3": "Mistral-7B-Instruct-v0.3",
    "Claude-3": "claude-3-haiku-20240307",
}

FED_METRICS = [
    "interesting", "engaging", "specific", "relevant", "correct",
    "semantically appropriate", "understandable", "fluent", "coherent",
    "error recovery", "consistent", "diverse", "depth", "likeable",
    "understand", "flexible", "informative", "inquisitive",
]
IXQUISITE_METRICS = [
    "minimal_explanations", "lexical_complexity", "synonym_density",
    "coherence", "reading_grade", "adaptation",
]

# Cluster grouping for the headline summary. Engagement metrics are the
# interaction-related aspects the paper argues LLMs are weak at; fluency
# metrics are the surface aspects the paper argues LLMs are already strong
# at and that the refinement trade-off is *expected* to push down.
ENGAGEMENT_CLUSTER = [
    "interesting", "engaging", "specific", "likeable",
    "flexible", "informative", "inquisitive",
]
FLUENCY_CLUSTER = ["semantically appropriate", "understandable", "fluent"]
OTHER_CLUSTER = [
    "relevant", "correct", "coherent", "error recovery",
    "consistent", "diverse", "depth", "understand",
]

ALPHA = 0.05


def cohens_d_paired(diff):
    s = diff.std(ddof=1)
    return 0.0 if s == 0 else diff.mean() / s


def per_metric(df, metric):
    task_col, tuned_col = metric, f"{metric}-tuned"
    if task_col not in df.columns or tuned_col not in df.columns:
        return None
    pair = df[[task_col, tuned_col]].dropna()
    if len(pair) < 2:
        return None
    t = pair[task_col]
    u = pair[tuned_col]
    diff = u - t
    if diff.std(ddof=1) == 0:
        return {"n": len(pair), "mean_diff": float(diff.mean()), "p": 1.0,
                "d": 0.0, "improved": False, "degraded": False}
    _, p = stats.ttest_rel(u, t)
    d = cohens_d_paired(diff)
    return {
        "n": len(pair),
        "mean_diff": float(diff.mean()),
        "p": float(p),
        "d": float(d),
        "improved": p < ALPHA and diff.mean() > 0,
        "degraded": p < ALPHA and diff.mean() < 0,
    }


def summarise_cell(dataset_folder, model_id):
    fname = f"{dataset_folder}_{model_id}_l30_w2_eval_tuned_eval.json"
    path = os.path.join(BASE_DIR, dataset_folder, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_json(path)
    out = {"file": fname, "rows": len(df), "fed": [], "ixq": []}
    for m in FED_METRICS:
        r = per_metric(df, m)
        if r is not None:
            out["fed"].append((m, r))
    for m in IXQUISITE_METRICS:
        r = per_metric(df, m)
        if r is not None:
            out["ixq"].append((m, r))
    return out


def fmt_count(metrics):
    imp = sum(1 for _, r in metrics if r["improved"])
    deg = sum(1 for _, r in metrics if r["degraded"])
    return imp, deg, len(metrics)


def cluster_counts(metrics, cluster_names):
    sub = [(m, r) for m, r in metrics if m in cluster_names]
    return fmt_count(sub)


def main():
    print("# Per-dataset significance: tuned vs. task (vanilla prompt)\n")
    print(f"Paired t-test, alpha = {ALPHA}. Each FED metric on the 0–100 scale per the paper's evaluation pipeline.\n")

    cells = []
    for dataset in DATASETS:
        for label, model_id in MODELS.items():
            cell = summarise_cell(dataset, model_id)
            if cell is not None:
                cells.append((dataset, label, cell))

    # ---- Headline: cluster-grouped ----
    print("## Headline (cluster-grouped)\n")
    print(
        "Engagement = interesting, engaging, specific, likeable, flexible, "
        "informative, inquisitive (7 metrics).\n"
        "Fluency = semantically appropriate, understandable, fluent (3 metrics).\n"
        "Other = relevant, correct, coherent, error recovery, consistent, "
        "diverse, depth, understand (8 metrics).\n"
    )
    print("| Dataset | Model | N | Engagement ↑ /7 | Fluency ↓ /3 | Other ↑ /8 | Other ↓ /8 | IXQ ↑ /6 | IXQ ↓ /6 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, label, cell in cells:
        eng_up, _, _ = cluster_counts(cell["fed"], ENGAGEMENT_CLUSTER)
        _, flu_dn, _ = cluster_counts(cell["fed"], FLUENCY_CLUSTER)
        oth_up, oth_dn, _ = cluster_counts(cell["fed"], OTHER_CLUSTER)
        ixq_up, ixq_dn, _ = fmt_count(cell["ixq"])
        print(f"| {DISPLAY[dataset]} | {label} | {cell['rows']} | "
              f"{eng_up} | {flu_dn} | {oth_up} | {oth_dn} | {ixq_up} | {ixq_dn} |")

    # ---- Raw FED counts (for transparency) ----
    print("\n## Raw FED ↑/↓ counts (full 18 metrics)\n")
    print("| Dataset | Model | N | FED ↑ | FED ↓ | FED total |")
    print("|---|---|---:|---:|---:|---:|")
    for dataset, label, cell in cells:
        fed_up, fed_dn, fed_tot = fmt_count(cell["fed"])
        print(f"| {DISPLAY[dataset]} | {label} | {cell['rows']} | "
              f"{fed_up} | {fed_dn} | {fed_tot} |")

    # ---- Per-metric details ----
    print("\n## Per-metric details (appendix)\n")
    for dataset, label, cell in cells:
        print(f"### {DISPLAY[dataset]} – {label} (N={cell['rows']})\n")
        print("| Metric | mean Δ | p | d | direction |")
        print("|---|---:|---:|---:|---|")
        for m, r in cell["fed"] + cell["ixq"]:
            arrow = "↑**" if r["improved"] else ("↓**" if r["degraded"] else "·")
            p_str = f"{r['p']:.1e}" if r["p"] < 0.001 else f"{r['p']:.3f}"
            print(f"| {m} | {r['mean_diff']:+.3f} | {p_str} | {r['d']:+.2f} | {arrow} |")
        print()


if __name__ == "__main__":
    main()
