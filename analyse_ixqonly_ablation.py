"""Compare the IXQuisite-only refinement ablation against the original
FED+IXQ-driven refinement and the un-tuned task baseline.

Three comparisons (paired t-test, p<0.05, on Claude × ELI5 vanilla):
  - task vs FED+IXQ-tuned (the paper's setup; baseline to confirm)
  - task vs IXQ-only-tuned (the ablation; the disentangle answer)
  - FED+IXQ-tuned vs IXQ-only-tuned (do the two refinement signals
    produce significantly different gains?)

If task vs IXQ-only-tuned shows the engagement cluster going up at
similar magnitude to task vs FED+IXQ-tuned, FED-derived feedback is
not necessary for the gains and the bias-vs-refinement confound is
empirically dissolved.
"""
import os
import pandas as pd
from scipy import stats

ORIGINAL_TUNED = "data/final_results/ELI5/ELI5_claude-3-haiku-20240307_l30_w2_eval_tuned_eval.json"
IXQ_TUNED = "data/evaluated_results/ELI5/ELI5_claude-3-haiku-20240307_l30_w2_eval_tuned_ixqonly_revisedwith_haiku45_eval.json"

ENGAGEMENT = ["interesting", "engaging", "specific", "likeable", "flexible", "informative", "inquisitive"]
FLUENCY = ["semantically appropriate", "understandable", "fluent"]
OTHER = ["relevant", "correct", "coherent", "error recovery", "consistent", "diverse", "depth", "understand"]
IXQUISITE = ["minimal_explanations", "lexical_complexity", "synonym_density", "coherence", "reading_grade", "adaptation"]

ALPHA = 0.05


def cohens_d_paired(diff):
    s = diff.std(ddof=1)
    return 0.0 if s == 0 else diff.mean() / s


def paired(df, col_a, col_b):
    pair = df[[col_a, col_b]].dropna()
    if len(pair) < 2:
        return None
    a = pair[col_a]
    b = pair[col_b]
    diff = b - a
    if diff.std(ddof=1) == 0:
        return {"n": len(pair), "mean_diff": 0.0, "p": 1.0, "d": 0.0,
                "improved": False, "degraded": False}
    _, p = stats.ttest_rel(b, a)
    d = cohens_d_paired(diff)
    return {
        "n": len(pair),
        "mean_diff": float(diff.mean()),
        "p": float(p),
        "d": float(d),
        "improved": p < ALPHA and diff.mean() > 0,
        "degraded": p < ALPHA and diff.mean() < 0,
    }


def cluster_summary(df, cluster_metrics, col_a_suffix, col_b_suffix, label):
    rows = []
    for m in cluster_metrics:
        col_a = f"{m}{col_a_suffix}"
        col_b = f"{m}{col_b_suffix}"
        if col_a not in df.columns or col_b not in df.columns:
            continue
        r = paired(df, col_a, col_b)
        if r is not None:
            rows.append((m, r))
    n_up = sum(1 for _, r in rows if r["improved"])
    n_down = sum(1 for _, r in rows if r["degraded"])
    avg_d = sum(r["d"] for _, r in rows) / max(len(rows), 1)
    avg_diff = sum(r["mean_diff"] for _, r in rows) / max(len(rows), 1)
    print(f"  {label:30s} {n_up:>2}↑ / {n_down:>2}↓ / {len(rows):>2} | mean Δ={avg_diff:+.3f} | mean |d|={avg_d:+.2f}")
    return rows


def per_metric_table(df, suffix_a, suffix_b, metrics, header):
    print(f"\n#### {header}\n")
    print("| Metric | mean Δ | p | d | direction |")
    print("|---|---:|---:|---:|---|")
    for m in metrics:
        r = paired(df, f"{m}{suffix_a}", f"{m}{suffix_b}")
        if r is None:
            continue
        arrow = "↑**" if r["improved"] else ("↓**" if r["degraded"] else "·")
        p_str = f"{r['p']:.1e}" if r["p"] < 0.001 else f"{r['p']:.3f}"
        print(f"| {m} | {r['mean_diff']:+.3f} | {p_str} | {r['d']:+.2f} | {arrow} |")


def main():
    if not os.path.exists(IXQ_TUNED):
        raise SystemExit(f"IXQuisite-only re-evaluation file not found: {IXQ_TUNED}")
    if not os.path.exists(ORIGINAL_TUNED):
        raise SystemExit(f"Original tuned file not found: {ORIGINAL_TUNED}")

    df_orig = pd.read_json(ORIGINAL_TUNED)
    df_ixq = pd.read_json(IXQ_TUNED)

    # align by (file, index) so paired tests compare same instances
    if 'file' in df_orig.columns and 'index' in df_orig.columns:
        df_orig['_key'] = df_orig['file'].astype(str) + "_" + df_orig['index'].astype(str)
        df_ixq['_key'] = df_ixq['file'].astype(str) + "_" + df_ixq['index'].astype(str)
        merged = df_orig.merge(
            df_ixq[['_key'] + [c for c in df_ixq.columns if c.endswith('-tuned')]],
            on='_key', how='inner', suffixes=('_orig', '_ixq'))
    else:
        # fallback: assume same row order
        merged = df_orig.copy()
        for c in df_ixq.columns:
            if c.endswith('-tuned'):
                merged[c + '_ixq'] = df_ixq[c].values

    # rename original *-tuned to *-fedtuned, ixq tuned to *-ixqtuned
    rename = {}
    for c in list(merged.columns):
        if c.endswith('-tuned') and not c.endswith('_orig') and not c.endswith('_ixq'):
            rename[c] = c.replace('-tuned', '-fedtuned')
        elif c.endswith('-tuned_orig'):
            rename[c] = c.replace('-tuned_orig', '-fedtuned')
        elif c.endswith('-tuned_ixq'):
            rename[c] = c.replace('-tuned_ixq', '-ixqtuned')
    merged = merged.rename(columns=rename)

    n_aligned = len(merged)
    n_orig = len(df_orig)
    n_ixq = len(df_ixq)
    n_ixq_done = df_ixq['engaging-tuned'].notna().sum() if 'engaging-tuned' in df_ixq.columns else 0
    print(f"# IXQuisite-only refinement ablation analysis\n")
    print(f"Original tuned file: {n_orig} rows; IXQ-only re-evaluated: {n_ixq_done}/{n_ixq} non-null tuned scores; aligned for pairing: {n_aligned}\n")

    print("## Cluster summary (paired t-test, p<{:.2f})\n".format(ALPHA))
    print("Comparison: TASK (un-tuned) vs FED+IXQ-tuned (paper's setup) vs IXQ-only-tuned (ablation).\n")
    print("```")
    print(f"  {'Cluster (n metrics)':30s} {'↑ / ↓ / N':>15s} | {'mean Δ':>10s} | {'mean d':>10s}")
    for cluster_name, metrics in [
        ("ENGAGEMENT (7)", ENGAGEMENT),
        ("FLUENCY (3)", FLUENCY),
        ("OTHER (8)", OTHER),
        ("IXQUISITE (6)", IXQUISITE),
    ]:
        print(f"\n[{cluster_name}]")
        cluster_summary(merged, metrics, "", "-fedtuned", "task → FED+IXQ-tuned")
        cluster_summary(merged, metrics, "", "-ixqtuned", "task → IXQ-only-tuned")
        cluster_summary(merged, metrics, "-fedtuned", "-ixqtuned", "FED+IXQ-tuned → IXQ-only-tuned")
    print("```\n")

    print("\n## Per-metric details\n")
    per_metric_table(merged, "", "-fedtuned", ENGAGEMENT + FLUENCY,
                     "task vs FED+IXQ-tuned (engagement + fluency)")
    per_metric_table(merged, "", "-ixqtuned", ENGAGEMENT + FLUENCY,
                     "task vs IXQ-only-tuned (engagement + fluency)")


if __name__ == "__main__":
    main()
