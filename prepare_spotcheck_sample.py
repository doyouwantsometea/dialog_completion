"""Prepare a 30-turn ReWIRED sample for manual explanation-bearing labelling.

Outputs a markdown file under paper/ with each sampled explainer turn
shown alongside its surrounding dialogue context. The user fills in
'yes' / 'no' / 'partial' on each Label: line, then runs
score_spotcheck.py to compute the rate.

Sampling: random.seed=42 for reproducibility, drawn from the *built*
ReWIRED data points (l=30, w=2) used in the paper's main experiments.
We sample from the union of explainer turns surviving in any model's
results file, so the sample is representative of the rows that actually
fed the rebuttal-relevant statistics.
"""
import os
import random
import pandas as pd

BASE_DIR = "data/final_results/WIRED"
N = 30
OUT = "paper/ARR_Rebuttal_Spotcheck_Sample.md"

random.seed(42)


def load_unique_turns():
    seen = set()
    rows = []
    for fname in sorted(os.listdir(BASE_DIR)):
        if not fname.endswith("_eval_tuned_eval.json"):
            continue
        df = pd.read_json(os.path.join(BASE_DIR, fname))
        for _, r in df.iterrows():
            key = (r.get("file"), r.get("index"))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "file": r.get("file"),
                "index": r.get("index"),
                "topic": r.get("topic"),
                "explainer": r.get("explainer"),
                "explainee": r.get("explainee"),
                "dialogue": r.get("dialogue"),
                "target_turn": r.get("target_turn"),
            })
    return rows


def main():
    rows = load_unique_turns()
    print(f"Total unique (file, index) explainer turns in ReWIRED: {len(rows)}")
    sample = random.sample(rows, N)

    with open(OUT, "w") as f:
        f.write("# Spot-check: are these ReWIRED explainer turns explanation-bearing?\n\n")
        f.write(
            "Reviewer jAec W2.2 doubts that every length-filtered explainer turn is "
            "strictly explanatory. To put a number on this for the rebuttal, label each "
            "of the 30 sampled turns below as **yes** / **partial** / **no** depending "
            "on whether the *target turn* (the one we'd ask an LLM to fill in) is "
            "explanation-bearing in the strict sense (concept / reason / evidence). "
            "Edit the `Label:` line for each item.\n\n"
        )
        f.write(f"Sample: N={N}, random.seed=42, drawn from "
                f"{len(rows)} unique ReWIRED explainer turns "
                f"(l=30, w=2 vanilla pipeline).\n\n")
        f.write("---\n\n")

        for i, r in enumerate(sample, 1):
            topic = r.get("topic")
            topic_str = topic.strip() if isinstance(topic, str) else ""
            f.write(f"## {i}. file={r['file']}, idx={r['index']}\n\n")
            if topic_str:
                f.write(f"_Topic context: {topic_str}_\n\n")
            f.write("**Dialogue context (with the target turn marked):**\n\n")
            ctx = r["dialogue"]
            tgt = (r["target_turn"] or "").strip()
            if tgt and tgt in ctx:
                ctx = ctx.replace(tgt, f"**[TARGET]** {tgt} **[/TARGET]**")
            else:
                # the {missing part} placeholder sits in dialogue; replace it
                ctx = ctx.replace(
                    "{missing part}",
                    f"**[TARGET]** {tgt} **[/TARGET]**" if tgt
                    else "**[TARGET — missing]**")
            f.write("```\n" + ctx.strip() + "\n```\n\n")
            f.write("Label: \n\n")
            f.write("---\n\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
