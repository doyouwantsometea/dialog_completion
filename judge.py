import asyncio
import json
import os
import pandas as pd
import random
from google import genai
from google.genai import types
from sklearn.model_selection import train_test_split
from tqdm.asyncio import tqdm_asyncio


# Seed Python's random so the A/B/C label shuffle in create_blind_prompt is
# reproducible across runs. (sklearn's train_test_split is seeded separately
# via random_state=42 and uses numpy, so it is unaffected.)
random.seed(42)


# ================= CONFIGURATION =================
with open('key.json', 'r') as f:
    API_KEY = json.loads(f.read())['Gemini']
MODEL_NAME = "gemini-2.5-pro"

# Path to your main results folder
BASE_DIR = "data/final_results"

# We want 100 samples from each dataset = 300 total
SAMPLES_PER_DATASET = 100

# Target the "Vanilla" files.
# We filter for files that DO contain 'eval_tuned_eval' but DO NOT contain 'openend' or 'topic'.
TARGET_SUFFIX = "eval_tuned_eval.json"
EXCLUDE_KEYWORDS = ["openend", "topic", "speakers"]

OUTPUT_FILE = "judgements/validation_results_incremental.jsonl"
SUMMARY_TABLE_FILE = "judgements/rebuttal_table.md"

# Strict Rate Limiting settings
CONCURRENCY_LIMIT = 1
REQUEST_DELAY = 4 # Seconds between requests


# ================= 1. DATA LOADING & SAMPLING =================
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
        filepath = os.path.join(folder_path, selected_file)

        with open(filepath, 'r') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
        elif isinstance(raw_data, dict):
            if "model_output" in raw_data or "task_output" in raw_data:
                df = pd.DataFrame(raw_data)
            else:
                df = pd.DataFrame.from_dict(raw_data, orient='index')
        else:
            continue

        rename_map = {'dialogue': 'dialogue_context', 'target_turn': 'original_turn', 'model_output': 'task_output'}
        df = df.rename(columns=rename_map)

        required_cols = ['dialogue_context', 'original_turn', 'task_output', 'tuned_output']
        if any(c not in df.columns for c in required_cols):
            continue

        # Create a unique ID for resumption
        # If 'file' and 'index' exist, use them. Else hash the context.
        if 'file' in df.columns and 'index' in df.columns:
            df['unique_id'] = df['file'].astype(str) + "_" + df['index'].astype(str)
        else:
            df['unique_id'] = df['dialogue_context'].apply(lambda x: str(hash(x)))

        df['dialogue_context'] = df['dialogue_context'].fillna("")
        df['len_cat'] = pd.qcut(df['dialogue_context'].str.len(), 3, labels=["short", "medium", "long"])

        try:
            sampled, _ = train_test_split(df, train_size=SAMPLES_PER_DATASET, stratify=df['len_cat'], random_state=42)
        except ValueError:
            sampled = df.sample(n=min(len(df), SAMPLES_PER_DATASET), random_state=42)

        sampled['dataset_source'] = dataset
        all_samples.extend(sampled.to_dict(orient='records'))

    return all_samples


# ================= 2. PROMPT & API =================

def create_blind_prompt(row):
    human = row.get('original_turn', '')
    task = row.get('task_output', '')
    tuned = row.get('tuned_output', '')

    options = [
        {"id": "human", "text": human},
        {"id": "task", "text": task},
        {"id": "tuned", "text": tuned}
    ]
    random.shuffle(options)
    option_map = {label: opt for label, opt in zip(['A', 'B', 'C'], options)}

    prompt_text = f"""
You are an expert evaluator of explanatory dialogue systems.
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

Output JSON ONLY:
{{ "best_option": "A", "worst_option": "C", "reasoning": "explanation" }}
"""
    return prompt_text, option_map


async def get_judgement_safe(client, row, semaphore):
    async with semaphore:
        await asyncio.sleep(REQUEST_DELAY)  # Rate limit safety

        retries = 0
        while retries < 3:
            try:
                prompt, option_map = create_blind_prompt(row)

                # --- CRITICAL FIX: USE .aio ACCESSOR ---
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )

                text_content = response.text.strip()
                if text_content.startswith("```"):
                    text_content = text_content.split("\n", 1)[1].rsplit("\n", 1)[0]

                result = json.loads(text_content)
                raw_choice = result.get('best_option', 'A')
                best_letter = str(raw_choice).strip().upper()[0]

                return {
                    "unique_id": row['unique_id'],
                    "dataset": row['dataset_source'],
                    "winner": option_map[best_letter]['id'],
                    "reasoning": result.get('reasoning', '')
                }

            except Exception as e:
                if "429" in str(e) or "503" in str(e):
                    await asyncio.sleep(5 * (retries + 1))
                    retries += 1
                else:
                    return {"unique_id": row['unique_id'], "error": str(e)}

        return {"unique_id": row['unique_id'], "error": "Max retries"}


# ================= 3. ORCHESTRATION =================

async def main():
    # 1. Load Todo List
    all_rows = load_and_sample_data()
    print(f"Loaded {len(all_rows)} targets.")

    # 2. Load Done List (Resume)
    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec['unique_id'])
                except:
                    pass
    print(f"Found {len(done_ids)} completed samples. Resuming...")

    # 3. Filter
    todo_rows = [r for r in all_rows if r['unique_id'] not in done_ids]
    print(f"Remaining to process: {len(todo_rows)}")

    if not todo_rows:
        generate_report()
        return

    # 4. Execute
    client = genai.Client(api_key=API_KEY)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_and_save(row):
        result = await get_judgement_safe(client, row, sem)
        # Save immediately
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
        return result

    # Show progress bar
    await tqdm_asyncio.gather(*[process_and_save(r) for r in todo_rows])

    # 5. Final Report
    generate_report()


def generate_report():
    if not os.path.exists(OUTPUT_FILE): return

    results = []
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty: return

    summary = df.groupby(['dataset', 'winner']).size().unstack(fill_value=0)
    summary['Total'] = summary.sum(axis=1)
    summary['% Human'] = (summary.get('human', 0) / summary['Total'] * 100).round(1)
    summary['% Tuned'] = (summary.get('tuned', 0) / summary['Total'] * 100).round(1)
    summary['% Task'] = (summary.get('task', 0) / summary['Total'] * 100).round(1)

    print("\n=== REBUTTAL TABLE ===")
    print(summary[['% Human', '% Task', '% Tuned']].to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
