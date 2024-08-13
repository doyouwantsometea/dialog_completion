import os
import pandas as pd
from tqdm import tqdm



df = pd.read_json('data/WikiDialog/data_validation.jsonl.gz', lines=True)

target_dir = 'data/WikiDialog/processed'
os.makedirs(target_dir, exist_ok=True)

print(df.head())


skipped_files = 0

for index, row in tqdm(df.iterrows(), total=len(df)):
    if len(row.author_num) < 10:
        skipped_files += 1
        continue
    # print(index)
    processed_df = pd.DataFrame(columns=['topic', 'dialog_lvl', 'role',
                                         'turn_num_tokens', 'turn'])
    
    for role, turn in zip(row.author_num, row.utterances):
        processed_df.at[len(processed_df), 'dialog_lvl'] = 'model'
        processed_df.at[len(processed_df)-1, 'role'] = 'Explainer' if role == 0 else 'Explainee'
        processed_df.at[len(processed_df)-1, 'turn'] = turn
        processed_df.at[len(processed_df)-1, 'turn_num_tokens'] = len(turn.split(' '))
        processed_df.at[len(processed_df)-1, 'topic'] = row.title
    
    processed_df.to_json(os.path.join(target_dir, f'WikiDialog_processed_{index}.json'))

print(f'Skipped {skipped_files} short dialogues.')