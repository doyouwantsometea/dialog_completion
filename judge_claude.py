"""Second-judge robustness check for the rebuttal (Edrb W2).

Mirrors judge.py exactly — same 300-turn stratified sample, same blind
3-way ranking prompt, same fixed seed for reproducibility — but uses
Anthropic Claude as the judge instead of Gemini. After both runs
complete, compute_judge_agreement.py reports per-item agreement and
the per-dataset preference table.

Model: claude-haiku-4-5-20251001 (latest, fastest, cheapest).
"""
import json
import os
import random
import time
import pandas as pd
from anthropic import Anthropic
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# match judge.py's seeding so the sampled rows and A/B/C labels are
# byte-identical to the Gemini run — required for inter-judge agreement
random.seed(42)

with open("key.json") as f:
    API_KEY = json.load(f)["Anthropic"]
MODEL_NAME = "claude-haiku-4-5-20251001"

BASE_DIR = "data/final_results"
SAMPLES_PER_DATASET = 100
TARGET_SUFFIX = "eval_tuned_eval.json"
EXCLUDE_KEYWORDS = ["openend", "topic", "speakers"]

OUTPUT_FILE = "judgements/validation_results_claude.jsonl"
REQUEST_DELAY = 1  # seconds; Claude Haiku tier-1 is fast enough


def load_and_sample_data():
    all_samples = []
    datasets = ["ELI5", "WikiDialog", "ReWIRED"]
    for dataset in datasets:
        folder_path = os.path.join(BASE_DIR, dataset)
        if not os.path.exists(folder_path) and dataset == "ReWIRED":
            folder_path = os.path.join(BASE_DIR, "WIRED")
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
        candidates = [f for f in os.listdir(folder_path) if f.endswith(TARGET_SUFFIX)]
        vanilla_files = [f for f in candidates if not any(k in f for k in EXCLUDE_KEYWORDS)]
        if not vanilla_files:
            continue
        selected_file = vanilla_files[0]
        with open(os.path.join(folder_path, selected_file)) as fh:
            raw_data = json.load(fh)
        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict):
            if "model_output" in raw_data or "task_output" in raw_data:
                df = pd.DataFrame(raw_data)
            else:
                df = pd.DataFrame.from_dict(raw_data, orient='index')
        else:
            continue
        rename_map = {'dialogue': 'dialogue_context', 'target_turn': 'original_turn',
                      'model_output': 'task_output'}
        df = df.rename(columns=rename_map)
        required_cols = ['dialogue_context', 'original_turn', 'task_output', 'tuned_output']
        if any(c not in df.columns for c in required_cols):
            continue
        if 'file' in df.columns and 'index' in df.columns:
            df['unique_id'] = df['file'].astype(str) + "_" + df['index'].astype(str)
        else:
            df['unique_id'] = df['dialogue_context'].apply(lambda x: str(hash(x)))
        df['dialogue_context'] = df['dialogue_context'].fillna("")
        df['len_cat'] = pd.qcut(df['dialogue_context'].str.len(), 3,
                                labels=["short", "medium", "long"])
        try:
            sampled, _ = train_test_split(df, train_size=SAMPLES_PER_DATASET,
                                          stratify=df['len_cat'], random_state=42)
        except ValueError:
            sampled = df.sample(n=min(len(df), SAMPLES_PER_DATASET), random_state=42)
        sampled['dataset_source'] = dataset
        all_samples.extend(sampled.to_dict(orient='records'))
    return all_samples


def create_blind_prompt(row):
    human = row.get('original_turn', '') or ''
    task = row.get('task_output', '') or ''
    tuned = row.get('tuned_output', '') or ''
    options = [
        {"id": "human", "text": human},
        {"id": "task", "text": task},
        {"id": "tuned", "text": tuned},
    ]
    random.shuffle(options)
    option_map = {label: opt for label, opt in zip(['A', 'B', 'C'], options)}
    prompt_text = f"""You are an expert evaluator of explanatory dialogue systems.
Task: Rank three candidate completions (A, B, C) for the missing 'Explainer' turn.
Prioritize deep conversational quality (SODA-EVAL criteria) over surface fluency.

1. **Interactivity:** Does it actively maintain the conversation?
2. **Clarity:** Is it accessible and jargon-free?
3. **Coherence:** Does it fit the logical flow?

Context:
{row.get('dialogue_context', '')}

Candidates:
[A]: {option_map['A']['text']}
[B]: {option_map['B']['text']}
[C]: {option_map['C']['text']}

Output JSON ONLY (no markdown, no commentary):
{{ "best_option": "A", "worst_option": "C", "reasoning": "explanation" }}
"""
    return prompt_text, option_map


def parse_json_response(text: str):
    """Strip optional markdown fences and parse JSON."""
    t = text.strip()
    if t.startswith("```"):
        # strip leading fence (with or without language tag)
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    return json.loads(t)


def get_judgement_safe(client, row):
    retries = 0
    while retries < 3:
        try:
            prompt, option_map = create_blind_prompt(row)
            msg = client.messages.create(
                model=MODEL_NAME,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text_content = msg.content[0].text
            result = parse_json_response(text_content)
            raw_choice = result.get('best_option', 'A')
            best_letter = str(raw_choice).strip().upper()[0]
            if best_letter not in option_map:
                raise ValueError(f"unparseable best_option: {raw_choice!r}")
            return {
                "unique_id": row['unique_id'],
                "dataset": row['dataset_source'],
                "winner": option_map[best_letter]['id'],
                "reasoning": result.get('reasoning', ''),
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "503" in err or "overloaded" in err.lower():
                time.sleep(5 * (retries + 1))
                retries += 1
            elif "json" in err.lower() or "Expecting" in err or "unparseable" in err:
                # parse error — retry once with fresh shuffle
                if retries == 0:
                    retries += 1
                    continue
                return {"unique_id": row['unique_id'], "error": err}
            else:
                return {"unique_id": row['unique_id'], "error": err}
    return {"unique_id": row['unique_id'], "error": "Max retries"}


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_rows = load_and_sample_data()
    print(f"Loaded {len(all_rows)} targets.")

    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec['unique_id'])
                except Exception:
                    pass
    print(f"Found {len(done_ids)} completed samples. Resuming...")
    todo = [r for r in all_rows if r['unique_id'] not in done_ids]
    print(f"Remaining: {len(todo)}")

    client = Anthropic(api_key=API_KEY)
    for row in tqdm(todo):
        time.sleep(REQUEST_DELAY)
        result = get_judgement_safe(client, row)
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

    # quick report
    print("\n# Claude judge per-dataset preference:\n")
    results = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    df = pd.DataFrame([r for r in results if "error" not in r])
    if not df.empty:
        summary = df.groupby(['dataset', 'winner']).size().unstack(fill_value=0)
        summary['Total'] = summary.sum(axis=1)
        for w in ['human', 'task', 'tuned']:
            if w in summary.columns:
                summary[f'% {w}'] = (summary[w] / summary['Total'] * 100).round(1)
        print(summary[[c for c in summary.columns if c.startswith('% ')]].to_markdown())


if __name__ == "__main__":
    main()
