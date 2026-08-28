"""Length-residualized FED re-scoring (Edrb W1 follow-up).

The FED-bias diagnostic in surface_feature_diagnostic.py shows that
FED scores on the *human reference* turn correlate strongly with four
surface features (token count, mean word length, punctuation density,
type-token ratio): the engagement cluster moves with longer / lower-TTR
turns, the fluency cluster with shorter / higher-TTR turns. Edrb's W1
asks whether the refinement gains we report are an artifact of FED
chasing those same surface features.

This script answers that question directly:

  1. Pool human reference turns across the 3 datasets (one model file
     per dataset to avoid double-counting). For each FED metric, fit
     an OLS regression of `metric_original ~ surface features` on the
     reference-only sample. This is the "bias model": the part of FED
     that is mechanically explained by surface features.

  2. Apply the same coefficients to *every* turn in every (dataset,
     model) cell — `model_output` (task) and `tuned_output` (tuned) —
     to obtain residualized FED scores: actual − predicted. The
     residual is the FED score with surface-feature variance removed.

  3. Per (dataset × model) cell, run a paired t-test on residualized
     `tuned − task` for each metric, with Cohen's d (paired). Aggregate
     by cluster (Engagement-7, Fluency-3, Other-8) and report the
     pooled-across-cells effect size.

If refinement gains are an artifact of the bias, residualization will
collapse them. If gains survive, they are not driven by the surface
features identified as the bias signature.
"""
import json
import os
import re
import string
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = "data/final_results"
DATASETS = {
    # display name → folder name
    "ReWIRED": "WIRED",
    "WikiDialog": "WikiDialog",
    "ELI5": "ELI5",
}
MODELS = [
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "claude-3-haiku-20240307",
]
FILE_TEMPLATE = "{dataset}_{model}_l30_w2_eval_tuned_eval.json"

FED_METRICS = [
    "interesting", "engaging", "specific", "relevant", "correct",
    "semantically appropriate", "understandable", "fluent", "coherent",
    "error recovery", "consistent", "diverse", "depth", "likeable",
    "understand", "flexible", "informative", "inquisitive",
]

# Cluster definitions match ARR_Rebuttal_General.md.
ENGAGEMENT = ["interesting", "engaging", "specific", "likeable",
              "flexible", "informative", "inquisitive"]
FLUENCY = ["semantically appropriate", "understandable", "fluent"]
OTHER = ["relevant", "correct", "coherent", "error recovery",
         "consistent", "diverse", "depth", "understand"]
assert set(ENGAGEMENT) | set(FLUENCY) | set(OTHER) == set(FED_METRICS)
assert len(ENGAGEMENT) + len(FLUENCY) + len(OTHER) == 18

SURFACE_COLS = ["len_tokens", "mean_word_len", "punct_density", "ttr"]


# ---------- surface features ----------

def surface_features(text) -> Dict[str, float]:
    if not isinstance(text, str) or not text.strip():
        return {c: np.nan for c in SURFACE_COLS}
    tokens = re.findall(r"\b\w+\b", text)
    n = len(tokens)
    if n == 0:
        return {"len_tokens": 0.0, "mean_word_len": np.nan,
                "punct_density": np.nan, "ttr": np.nan}
    word_lens = [len(t) for t in tokens]
    n_punct = sum(1 for c in text if c in string.punctuation)
    types = len(set(t.lower() for t in tokens))
    return {
        "len_tokens": float(n),
        "mean_word_len": float(np.mean(word_lens)),
        "punct_density": n_punct / max(len(text), 1),
        "ttr": types / n,
    }


def add_surface_features(df: pd.DataFrame, text_col: str, prefix: str) -> pd.DataFrame:
    feats = pd.DataFrame([surface_features(t) for t in df[text_col]])
    feats.columns = [f"{prefix}_{c}" for c in feats.columns]
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


# ---------- regression ----------

def fit_bias_model(ref_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Fit OLS metric_original ~ [1, surface] on human-reference rows.
    Returns {metric: coef vector of length 5 (intercept + 4 features)}.
    """
    X = np.column_stack([
        np.ones(len(ref_df)),
        ref_df[[f"ref_{c}" for c in SURFACE_COLS]].to_numpy(),
    ])
    coefs = {}
    for m in FED_METRICS:
        y = ref_df[f"{m}-original"].to_numpy()
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < 100:
            coefs[m] = None
            continue
        Xm = X[mask]
        ym = y[mask]
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        coefs[m] = beta
    return coefs


def predict(coefs: np.ndarray, surf: pd.DataFrame) -> np.ndarray:
    X = np.column_stack([np.ones(len(surf)), surf.to_numpy()])
    return X @ coefs


def cohens_d_paired(diff: np.ndarray) -> float:
    s = np.std(diff, ddof=1)
    return float(np.mean(diff) / s) if s > 0 else 0.0


# ---------- loaders ----------

def load_cell(dataset_folder: str, model: str) -> pd.DataFrame:
    path = os.path.join(BASE_DIR, dataset_folder,
                        FILE_TEMPLATE.format(dataset=dataset_folder, model=model))
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def build_pooled_reference(per_cell: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Take one model per dataset for the reference-fitting pool."""
    parts = []
    for display, folder in DATASETS.items():
        df = per_cell[(display, MODELS[0])].copy()  # Llama
        df = add_surface_features(df, "target_turn", "ref")
        keep = (
            [f"ref_{c}" for c in SURFACE_COLS]
            + [f"{m}-original" for m in FED_METRICS]
        )
        sub = df[keep].copy()
        sub["dataset"] = display
        parts.append(sub)
    pooled = pd.concat(parts, ignore_index=True)
    return pooled


def annotate_cell(df: pd.DataFrame, coefs: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = add_surface_features(df, "model_output", "task")
    df = add_surface_features(df, "tuned_output", "tuned")
    for m in FED_METRICS:
        beta = coefs[m]
        if beta is None:
            df[f"{m}-task-resid"] = np.nan
            df[f"{m}-tuned-resid"] = np.nan
            continue
        task_pred = predict(beta, df[[f"task_{c}" for c in SURFACE_COLS]])
        tuned_pred = predict(beta, df[[f"tuned_{c}" for c in SURFACE_COLS]])
        df[f"{m}-task-resid"] = df[m].to_numpy() - task_pred
        df[f"{m}-tuned-resid"] = df[f"{m}-tuned"].to_numpy() - tuned_pred
    return df


# ---------- per-cell tests ----------

def cell_results(df: pd.DataFrame, raw: bool = False) -> Dict[str, Tuple[float, float, int]]:
    """Returns {metric: (mean_diff, p_value, d, n)} for tuned vs task."""
    out = {}
    for m in FED_METRICS:
        if raw:
            a = df[m].to_numpy()
            b = df[f"{m}-tuned"].to_numpy()
        else:
            a = df[f"{m}-task-resid"].to_numpy()
            b = df[f"{m}-tuned-resid"].to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 30:
            out[m] = (np.nan, np.nan, np.nan, int(mask.sum()))
            continue
        diff = b[mask] - a[mask]
        if np.std(diff, ddof=1) == 0:
            out[m] = (float(np.mean(diff)), 1.0, 0.0, int(mask.sum()))
            continue
        t, p = stats.ttest_rel(b[mask], a[mask])
        d = cohens_d_paired(diff)
        out[m] = (float(np.mean(diff)), float(p), float(d), int(mask.sum()))
    return out


def cluster_summary(per_metric: Dict[str, Tuple[float, float, int]]):
    out = {}
    for cname, metrics in [("Engagement", ENGAGEMENT),
                           ("Fluency", FLUENCY),
                           ("Other", OTHER)]:
        n_up = 0
        n_down = 0
        n_sig_up = 0
        n_sig_down = 0
        ds = []
        for m in metrics:
            md, p, d, n = per_metric[m]
            if not np.isfinite(d):
                continue
            ds.append(d)
            if d > 0:
                n_up += 1
                if p < 0.05:
                    n_sig_up += 1
            elif d < 0:
                n_down += 1
                if p < 0.05:
                    n_sig_down += 1
        out[cname] = {
            "n_total": len(metrics),
            "n_up": n_up,
            "n_down": n_down,
            "n_sig_up": n_sig_up,
            "n_sig_down": n_sig_down,
            "mean_d": float(np.mean(ds)) if ds else np.nan,
        }
    return out


# ---------- driver ----------

def main():
    # 1. Load all 9 cells.
    per_cell: Dict[Tuple[str, str], pd.DataFrame] = {}
    for display, folder in DATASETS.items():
        for model in MODELS:
            try:
                per_cell[(display, model)] = load_cell(folder, model)
            except FileNotFoundError as e:
                print(f"# WARN: missing {display}/{model}: {e}")

    # 2. Fit bias model on pooled human-reference turns (one model per dataset).
    ref_df = build_pooled_reference(per_cell)
    print(f"# Pooled reference fit: N={len(ref_df)} turns "
          f"(ReWIRED={int((ref_df['dataset']=='ReWIRED').sum())}, "
          f"WikiDialog={int((ref_df['dataset']=='WikiDialog').sum())}, "
          f"ELI5={int((ref_df['dataset']=='ELI5').sum())})")
    coefs = fit_bias_model(ref_df)
    print("\n## Bias-model coefficients (FED_metric ~ 1 + len + meanwordlen + punctdensity + ttr)\n")
    print("| Metric | β₀ | β_len | β_meanwordlen | β_punctdens | β_ttr |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in FED_METRICS:
        b = coefs[m]
        if b is None:
            print(f"| {m} | – | – | – | – | – |")
        else:
            print(f"| {m} | {b[0]:+.3f} | {b[1]:+.4f} | {b[2]:+.3f} | {b[3]:+.3f} | {b[4]:+.3f} |")

    # 3. Annotate each (dataset, model) cell with residualized scores.
    annotated: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, df in per_cell.items():
        annotated[key] = annotate_cell(df, coefs)

    # 4. Per-cell raw and residualized cluster summaries.
    print("\n## Per-cell paired t-test on tuned − task (raw vs. residualized)\n")
    print("| Dataset | Model | N | Engage ↑/7 (raw) | Engage ↑/7 (resid) | Fluency ↓/3 (raw) | Fluency ↓/3 (resid) | Engage mean d (raw → resid) | Fluency mean d (raw → resid) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    pooled_diffs = {m: {"task": [], "tuned": []} for m in FED_METRICS}
    pooled_resid = {m: {"task": [], "tuned": []} for m in FED_METRICS}
    for display in DATASETS:
        for model in MODELS:
            key = (display, model)
            if key not in annotated:
                continue
            df = annotated[key]
            raw = cell_results(df, raw=True)
            resid = cell_results(df, raw=False)
            raw_summ = cluster_summary(raw)
            resid_summ = cluster_summary(resid)
            n = max(raw[m][3] for m in FED_METRICS)
            print(f"| {display} | {model.split('-')[0]} | {n} "
                  f"| {raw_summ['Engagement']['n_sig_up']} "
                  f"| {resid_summ['Engagement']['n_sig_up']} "
                  f"| {raw_summ['Fluency']['n_sig_down']} "
                  f"| {resid_summ['Fluency']['n_sig_down']} "
                  f"| {raw_summ['Engagement']['mean_d']:+.3f} → {resid_summ['Engagement']['mean_d']:+.3f} "
                  f"| {raw_summ['Fluency']['mean_d']:+.3f} → {resid_summ['Fluency']['mean_d']:+.3f} |")
            for m in FED_METRICS:
                a = df[m].to_numpy()
                b = df[f"{m}-tuned"].to_numpy()
                mask = np.isfinite(a) & np.isfinite(b)
                pooled_diffs[m]["task"].extend(a[mask].tolist())
                pooled_diffs[m]["tuned"].extend(b[mask].tolist())
                ar = df[f"{m}-task-resid"].to_numpy()
                br = df[f"{m}-tuned-resid"].to_numpy()
                maskr = np.isfinite(ar) & np.isfinite(br)
                pooled_resid[m]["task"].extend(ar[maskr].tolist())
                pooled_resid[m]["tuned"].extend(br[maskr].tolist())

    # 5. Pooled-across-9-cells per-metric summary.
    print("\n## Pooled-across-cells per-metric: tuned − task (raw and residualized)\n")
    print("| Metric | Cluster | N | mean Δ (raw) | d (raw) | p (raw) | mean Δ (resid) | d (resid) | p (resid) | shrinkage |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    cluster_of = {}
    for m in ENGAGEMENT: cluster_of[m] = "Engagement"
    for m in FLUENCY: cluster_of[m] = "Fluency"
    for m in OTHER: cluster_of[m] = "Other"

    cluster_d_raw = {"Engagement": [], "Fluency": [], "Other": []}
    cluster_d_res = {"Engagement": [], "Fluency": [], "Other": []}
    for m in FED_METRICS:
        a = np.array(pooled_diffs[m]["task"])
        b = np.array(pooled_diffs[m]["tuned"])
        diff = b - a
        d_raw = cohens_d_paired(diff)
        t_raw, p_raw = stats.ttest_rel(b, a)
        ar = np.array(pooled_resid[m]["task"])
        br = np.array(pooled_resid[m]["tuned"])
        diffr = br - ar
        d_res = cohens_d_paired(diffr)
        t_res, p_res = stats.ttest_rel(br, ar)
        n = len(a)
        shrink = (1 - abs(d_res) / abs(d_raw)) * 100 if abs(d_raw) > 1e-9 else 0
        cluster_d_raw[cluster_of[m]].append(d_raw)
        cluster_d_res[cluster_of[m]].append(d_res)
        print(f"| {m} | {cluster_of[m]} | {n} | "
              f"{np.mean(diff):+.3f} | {d_raw:+.3f} | {p_raw:.2e} | "
              f"{np.mean(diffr):+.3f} | {d_res:+.3f} | {p_res:.2e} | "
              f"{shrink:+.0f}% |")

    print("\n## Pooled cluster summary (mean Cohen's d across metrics in cluster)\n")
    print("| Cluster | mean d (raw) | mean d (resid) | retention |")
    print("|---|---:|---:|---:|")
    for c in ["Engagement", "Fluency", "Other"]:
        d_raw = np.mean(cluster_d_raw[c])
        d_res = np.mean(cluster_d_res[c])
        ret = (abs(d_res) / abs(d_raw)) * 100 if abs(d_raw) > 1e-9 else 0
        print(f"| {c} | {d_raw:+.3f} | {d_res:+.3f} | {ret:.0f}% |")


if __name__ == "__main__":
    main()
