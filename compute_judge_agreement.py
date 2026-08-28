"""Inter-judge agreement between Gemini and Claude on the 300-turn rebuttal sample.

Reads judgements/validation_results_incremental.jsonl (Gemini) and
judgements/validation_results_claude.jsonl (Claude), aligns by unique_id,
and reports:
- per-judge per-dataset preference table
- per-item agreement rate
- Cohen's kappa across the 3-way category {human, task, tuned}
- agreement on "human is best" yes/no (the headline claim)
"""
import json
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

GEMINI = "judgements/validation_results_incremental.jsonl"
CLAUDE = "judgements/validation_results_claude.jsonl"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def per_judge_summary(name, rows):
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        print(f"\n## {name}: no valid rows yet\n")
        return df
    print(f"\n## {name}: per-dataset preference (N total = {len(df)})\n")
    summary = df.groupby(['dataset', 'winner']).size().unstack(fill_value=0)
    summary['Total'] = summary.sum(axis=1)
    for w in ['human', 'task', 'tuned']:
        if w not in summary.columns:
            summary[w] = 0
        summary[f'% {w}'] = (summary[w] / summary['Total'] * 100).round(1)
    print(summary[['Total'] + [f'% {w}' for w in ['human', 'task', 'tuned']]].to_markdown())
    return df


def main():
    g = load_jsonl(GEMINI)
    c = load_jsonl(CLAUDE)
    print(f"Gemini rows: {len(g)} (errors: {sum('error' in r for r in g)})")
    print(f"Claude rows: {len(c)} (errors: {sum('error' in r for r in c)})")

    df_g = per_judge_summary("Gemini-2.5-Pro", g)
    df_c = per_judge_summary("Claude-Haiku-4.5", c)

    if df_g.empty or df_c.empty:
        print("\n(skip agreement — one or both judges incomplete)")
        return

    merged = df_g.merge(df_c, on='unique_id', suffixes=('_g', '_c'), how='inner')
    print(f"\n## Inter-judge agreement (N aligned = {len(merged)})\n")

    same = (merged['winner_g'] == merged['winner_c']).sum()
    rate = 100.0 * same / len(merged)
    kappa = cohen_kappa_score(merged['winner_g'], merged['winner_c'])
    print(f"- 3-way exact agreement: **{same}/{len(merged)} ({rate:.1f}%)**")
    print(f"- Cohen's κ (3-way): **{kappa:.3f}**")

    # Headline: agreement on "human is best"
    h_g = (merged['winner_g'] == 'human')
    h_c = (merged['winner_c'] == 'human')
    both_human = (h_g & h_c).sum()
    only_g = (h_g & ~h_c).sum()
    only_c = (~h_g & h_c).sum()
    neither = (~h_g & ~h_c).sum()
    binary_kappa = cohen_kappa_score(h_g, h_c)
    print(f"\n### 'Human is best' binarisation\n")
    print(f"|  | Claude: human | Claude: not human |")
    print(f"|---|---:|---:|")
    print(f"| Gemini: human | {both_human} | {only_g} |")
    print(f"| Gemini: not human | {only_c} | {neither} |")
    print(f"\nBinary κ on 'human-best': **{binary_kappa:.3f}**")

    # Per-dataset breakdown
    print(f"\n### Per-dataset cross-judge preference\n")
    print("| Dataset | N | Gemini % human | Claude % human | Gemini % task | Claude % task | Gemini % tuned | Claude % tuned |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds, sub in merged.groupby('dataset_g'):
        n = len(sub)
        gg = (sub['winner_g'] == 'human').sum() / n * 100
        cc = (sub['winner_c'] == 'human').sum() / n * 100
        gt = (sub['winner_g'] == 'task').sum() / n * 100
        ct = (sub['winner_c'] == 'task').sum() / n * 100
        gu = (sub['winner_g'] == 'tuned').sum() / n * 100
        cu = (sub['winner_c'] == 'tuned').sum() / n * 100
        print(f"| {ds} | {n} | {gg:.1f} | {cc:.1f} | {gt:.1f} | {ct:.1f} | {gu:.1f} | {cu:.1f} |")


if __name__ == "__main__":
    main()
