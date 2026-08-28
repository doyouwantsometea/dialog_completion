"""IXQuisite-only refinement ablation: Step 2 — re-evaluate tuned outputs.

Reads the tuned file produced by run_ixqonly_tuning.py, scores every
non-null tuned_output with FED + IXQuisite, and writes the result in the
same column convention as the production pipeline (`*-tuned` columns).
The output is plug-compatible with paper/rebuttal_stats.py.

FED runs on whatever accelerator fed.py auto-detects (CUDA → MPS → CPU).
DialoGPT-large is small (~774M params), so even CPU completes in
~10–20 minutes for the ~1200-instance ELI5 cell.
"""
import os
import pandas as pd
from tqdm import tqdm

import fed
from IXQuisite import IXQuisite
from utils import flatten_dialogue

INPUT_FILE = "data/tuned_results/ELI5/ELI5_claude-3-haiku-20240307_l30_w2_eval_tuned_ixqonly_revisedwith_haiku45.json"
OUTPUT_FILE = "data/evaluated_results/ELI5/ELI5_claude-3-haiku-20240307_l30_w2_eval_tuned_ixqonly_revisedwith_haiku45_eval.json"

# FED returns these 18 metrics; their "*-tuned" column scaling matches
# evaluation.py: metrics in MUL_BY_100 are multiplied by 100, the rest are
# stored raw. Replicating that convention keeps the file plug-compatible
# with the production stats pipeline.
FED_METRICS = [
    "interesting", "engaging", "specific", "relevant", "correct",
    "semantically appropriate", "understandable", "fluent", "coherent",
    "error recovery", "consistent", "diverse", "depth", "likeable",
    "understand", "flexible", "informative", "inquisitive",
]
MUL_BY_100 = {
    "interesting", "engaging", "specific", "semantically appropriate",
    "understandable", "fluent", "likeable", "flexible", "informative",
    "inquisitive",
}

IXQ_METRICS = [
    "minimal_explanations", "lexical_complexity", "synonym_density",
    "coherence", "reading_grade", "adaptation",
]


def fed_score_to_col(metric, raw):
    val = raw * 100 if metric in MUL_BY_100 else raw
    return round(val, 4)


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Tuned file not found: {INPUT_FILE}. Run run_ixqonly_tuning.py first.")

    df = pd.read_json(INPUT_FILE)
    n_total = len(df)
    n_tuned = df['tuned_output'].apply(lambda x: isinstance(x, str) and bool(x)).sum()
    print(f"Loaded {n_total} rows; {n_tuned} have valid tuned_output to score.")

    # initialise the *-tuned columns so dtype is consistent
    new_cols = [f"{m}-tuned" for m in FED_METRICS + IXQ_METRICS]
    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    # resume support: skip rows that already have FED tuned scores
    if 'engaging-tuned' in df.columns:
        already_done = df['engaging-tuned'].notna().sum()
        if already_done > 0:
            print(f"Resuming: {already_done} rows already evaluated.")

    print(f"Loading FED model on device={fed.device}...")
    fed_model, fed_tokenizer = fed.load_models('microsoft/DialoGPT-large')
    print("FED model loaded.")

    save_every = 25
    since_save = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        if not isinstance(row.tuned_output, str) or not row.tuned_output:
            continue
        # already-evaluated rows
        if pd.notna(df.at[index, 'engaging-tuned']):
            continue

        try:
            # FED
            conversation = flatten_dialogue(
                dialogue=row.dialogue,
                reference=row.target_turn,
                model_turn=row.tuned_output,
                original_dialog=False,
            )
            scores = fed.evaluate(conversation, fed_model, fed_tokenizer)
            for m in FED_METRICS:
                df.at[index, f"{m}-tuned"] = fed_score_to_col(m, scores[m])

            # IXQuisite
            ts = IXQuisite(datapoint=row.to_dict(),
                           original_dialog=False,
                           tuned=True, r=4)
            ixq_scores = ts.get_scores()
            for m in IXQ_METRICS:
                df.at[index, f"{m}-tuned"] = ixq_scores[m]

        except Exception as e:
            print(f"\n[row {index}] eval error: {e}")
            continue

        since_save += 1
        if since_save >= save_every:
            df.to_json(OUTPUT_FILE)
            since_save = 0

    df.to_json(OUTPUT_FILE)
    n_done = df['engaging-tuned'].notna().sum()
    print(f"\nDone. {n_done}/{n_total} rows evaluated. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    main()
