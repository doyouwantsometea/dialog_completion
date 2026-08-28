"""IXQuisite-only refinement ablation: Step 1 — generate tuned outputs.

Disentangles whether self-refinement gains are an artifact of FED bias
(reviewer Edrb W1, jAec W1, ckDa W2) by re-running the refinement
pipeline using ONLY the 6 IXQuisite features to pick worst features —
no FED signal in the loop. If engagement metrics still go up after
re-evaluation, FED-derived feedback is not necessary for the gains.

Cell: Claude-3-Haiku × ELI5 vanilla. Saves tuned_output incrementally
so the run is restartable.
"""
import os
import time
import pandas as pd
from tqdm import tqdm

from prompter import Prompter
from utils import extract_json, get_worst_features
from instruct_tuning import (
    process_feature_stat,
    feature_to_description,
    get_instructions,
)

# ---- config ----
INPUT_FILE = "data/evaluated_results/ELI5/ELI5_claude-3-haiku-20240307_l30_w2_eval.json"
# Note on revising model: the paper used claude-3-haiku-20240307 for both
# initial generation AND refinement. That model was deprecated by Anthropic
# in 2025, so this ablation uses claude-haiku-4-5-20251001 to revise the
# (still claude-3-haiku-20240307) baseline outputs. The disentangle
# argument is about *signal* (FED+IXQ vs IXQ-only feedback), not about
# which LLM does the revision.
OUTPUT_FILE = "data/tuned_results/ELI5/ELI5_claude-3-haiku-20240307_l30_w2_eval_tuned_ixqonly_revisedwith_haiku45.json"
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 256  # paper's default is 100; bumped to 256 to avoid truncated JSON from Haiku 4.5 (which sometimes wraps in markdown fences)
N_FEATURES = 3

IXQ_FEATURES = [
    "minimal_explanations", "lexical_complexity", "synonym_density",
    "coherence", "reading_grade", "adaptation",
]


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        print(f"Resuming from {OUTPUT_FILE}")
        df = pd.read_json(OUTPUT_FILE)
    else:
        print(f"Loading {INPUT_FILE}")
        df = pd.read_json(INPUT_FILE)
        # populate IXQ-only worst features
        for feature in IXQ_FEATURES:
            process_feature_stat(df, feature)
        ixq_dif_cols = [f"{f}-dif" for f in IXQ_FEATURES]
        df['worst_features'] = df[ixq_dif_cols].apply(
            get_worst_features, n=N_FEATURES, axis=1)
        df['tuned_output'] = None

    print(f"Total rows: {len(df)}")
    print(f"Already tuned: {df['tuned_output'].notna().sum()}")
    print(f"Skipped (no model_output): {df['model_output'].isna().sum()}")

    # patch instructions module so feature_to_description sees them
    import instruct_tuning
    instruct_tuning.instructions = get_instructions()

    prompter = Prompter(prompt_cfg_filename='prompts.json', task='tuning')

    # Bypass AnthropicModelLoader to use a higher max_tokens than the
    # codebase default of 100 — Haiku 4.5 sometimes pads JSON with markdown
    # fences which can blow past 100.
    import json as _json
    from anthropic import Anthropic
    api_key = _json.load(open('key.json'))['Anthropic']
    client = Anthropic(api_key=api_key)

    def call_llm(prompt_text: str) -> str:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return msg.content[0].text

    save_every = 25
    since_save = 0
    api_errors = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        if pd.notna(row.tuned_output):
            continue
        if not isinstance(row.model_output, str) or not row.model_output:
            continue
        if not row.worst_features or len(row.worst_features) == 0:
            continue

        dialogue = row.dialogue.replace(
            '{missing part}',
            f'<model-generated> {row.model_output} </model-generated>')
        description = feature_to_description(
            worst_features=row.worst_features, original_prompt=False)
        prompt = prompter.build_prompt(
            dialogue=dialogue, instruction=description)

        try:
            raw_output = call_llm(prompt)
        except Exception as e:
            api_errors += 1
            print(f"\n[row {index}] API error: {e}")
            time.sleep(5)
            continue

        json_output = extract_json(raw_output)
        if not json_output:
            continue
        tuned = json_output.get('revised turn', None)
        df.at[index, 'tuned_output'] = tuned

        since_save += 1
        if since_save >= save_every:
            df.to_json(OUTPUT_FILE)
            since_save = 0

    df.to_json(OUTPUT_FILE)
    print(f"\nDone. API errors: {api_errors}. "
          f"Final tuned count: {df['tuned_output'].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    main()
