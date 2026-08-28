"""Surface-feature regression diagnostic for the FED bias claim.

Hypothesis (from paper §5.2): FED prefers written-style turns. If true,
FED scores on the *human reference* should correlate with surface
features that distinguish written from spoken language — turn length,
mean word length, punctuation density, type-token ratio.

For each dataset we score every human reference turn on these surface
features, then correlate (Pearson r) the surface features with the
FED scores on the same human reference (the `*-original` columns). A
strong positive correlation between, e.g., length and the `engaging`
score across spoken+written turns is direct evidence that FED conflates
written-style polish with conversational quality.
"""
import os
import re
import string
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = "data/final_results"
DATASETS = {
    "ReWIRED": "WIRED",
    "WikiDialog": "WikiDialog",
    "ELI5": "ELI5",
}
# Use Llama files for the per-dataset sample (highest task accomplishment
# rate across datasets, per Table 1 of the paper). target_turn and
# *-original columns are identical across models for the same dataset, so
# the choice of model only affects which subset of the original turns is
# represented.
FILE_TEMPLATE = "{prefix}_Meta-Llama-3.1-8B-Instruct_l30_w2_eval_tuned_eval.json"

FED_METRICS = [
    "interesting", "engaging", "specific", "relevant", "correct",
    "semantically appropriate", "understandable", "fluent", "coherent",
    "error recovery", "consistent", "diverse", "depth", "likeable",
    "understand", "flexible", "informative", "inquisitive",
]
ORIGINAL_COLS = [f"{m}-original" for m in FED_METRICS]


def surface_features(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"len_tokens": np.nan, "mean_word_len": np.nan,
                "punct_density": np.nan, "ttr": np.nan}
    tokens = re.findall(r"\b\w+\b", text)
    n = len(tokens)
    if n == 0:
        return {"len_tokens": 0, "mean_word_len": np.nan,
                "punct_density": np.nan, "ttr": np.nan}
    word_lens = [len(t) for t in tokens]
    n_punct = sum(1 for c in text if c in string.punctuation)
    types = len(set(t.lower() for t in tokens))
    return {
        "len_tokens": n,
        "mean_word_len": float(np.mean(word_lens)),
        "punct_density": n_punct / max(len(text), 1),
        "ttr": types / n,
    }


def load_dataset(display: str, folder: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, folder, FILE_TEMPLATE.format(prefix=folder))
    df = pd.read_json(path)
    sf = pd.DataFrame([surface_features(t) for t in df["target_turn"]])
    out = pd.concat([sf, df[ORIGINAL_COLS]], axis=1).dropna()
    out["dataset"] = display
    return out


def correlate(df: pd.DataFrame, surface_col: str, fed_col: str):
    pair = df[[surface_col, fed_col]].dropna()
    if len(pair) < 30:
        return None
    r, p = stats.pearsonr(pair[surface_col], pair[fed_col])
    return float(r), float(p), len(pair)


def main():
    parts = []
    for display, folder in DATASETS.items():
        try:
            df = load_dataset(display, folder)
            parts.append(df)
            print(f"# {display}: N={len(df)} reference turns "
                  f"(mean len={df['len_tokens'].mean():.1f}, "
                  f"mean word len={df['mean_word_len'].mean():.2f})")
        except FileNotFoundError as e:
            print(f"# WARN: {display} skipped — {e}")
    pooled = pd.concat(parts, ignore_index=True)

    surface_cols = ["len_tokens", "mean_word_len", "punct_density", "ttr"]
    show_metrics = [
        "engaging", "interesting", "likeable", "informative",
        "fluent", "understandable",
    ]

    # ---- Per-dataset and pooled correlations ----
    print("\n## Pearson r between surface features and FED-on-human-reference")
    print("\n(positive r = the surface feature predicts a higher FED score on the "
          "human turn; ★ marks |r|≥0.20.)\n")

    for display in list(DATASETS.keys()) + ["POOLED"]:
        print(f"\n### {display}\n")
        if display == "POOLED":
            df = pooled
        else:
            df = pooled[pooled["dataset"] == display]
        if len(df) < 30:
            print(f"(skip — N={len(df)} too small)")
            continue
        header = "| Metric | " + " | ".join(surface_cols) + " |"
        sep = "|---|" + "|".join(["---:"] * len(surface_cols)) + "|"
        print(header)
        print(sep)
        for m in show_metrics:
            cells = []
            for s in surface_cols:
                res = correlate(df, s, f"{m}-original")
                if res is None:
                    cells.append("–")
                else:
                    r, p, n = res
                    star = "★" if abs(r) >= 0.20 else ""
                    sig = "**" if p < 0.001 else ("*" if p < 0.05 else "")
                    cells.append(f"{r:+.2f}{star}{sig}")
            print(f"| {m} | " + " | ".join(cells) + " |")

    # ---- Direct dataset-mean comparison (the paper's headline bias claim) ----
    print("\n## Dataset-level means (anchors the bias claim)\n")
    print("| Dataset | N | len_tokens | mean_word_len | punct_density | ttr |")
    print("|---|---:|---:|---:|---:|---:|")
    for display in DATASETS:
        df = pooled[pooled["dataset"] == display]
        print(f"| {display} | {len(df)} | "
              f"{df['len_tokens'].mean():.1f} | "
              f"{df['mean_word_len'].mean():.2f} | "
              f"{df['punct_density'].mean():.4f} | "
              f"{df['ttr'].mean():.3f} |")

    print("\n| Dataset | N | engaging | interesting | likeable | fluent | understandable |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for display in DATASETS:
        df = pooled[pooled["dataset"] == display]
        cells = []
        for m in ["engaging", "interesting", "likeable", "fluent", "understandable"]:
            cells.append(f"{df[f'{m}-original'].mean():.2f}")
        print(f"| {display} | {len(df)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
