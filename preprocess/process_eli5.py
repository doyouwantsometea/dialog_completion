import os
import pandas as pd
from tqdm import tqdm
from random import sample


# id_1 = 10
# id_2 = 40

ids = [(6, 7),
       (6, 10),
       (6, 40),
       (6, 50),
       (8, 50),
       (10, 40)]

target_dir = 'data_resample/ELI5/processed'
os.makedirs(target_dir, exist_ok=True)

for id_1, id_2 in ids:

    df = pd.read_json(f'data/ELI5/label-studio-annotation-task-{id_1}-{id_2}.json')


    # print(df.head())


    # skipped_files = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):

        # print(index)
        processed_df = pd.DataFrame(columns=['topic', 'dialog_lvl', 'role',
                                            'turn_num_tokens', 'turn'])
        
        for turn in row.dialogue:
            # print(turn)
            processed_df.at[len(processed_df), 'dialog_lvl'] = 'eli5'
            processed_df.at[len(processed_df)-1, 'role'] = turn['author'].capitalize()
            processed_df.at[len(processed_df)-1, 'turn'] = turn['text'].replace('\n', ' ')
            processed_df.at[len(processed_df)-1, 'turn_num_tokens'] = len(turn['text'].split(' '))
            processed_df.at[len(processed_df)-1, 'topic'] = 'N/A'
        
        if (processed_df['turn_num_tokens'] > 200).any():
            continue
            # print(index)
        # print(processed_df.head())
        processed_df.to_json(os.path.join(target_dir, f'ELI5_processed_{id_1}{id_2}{index}.json'))


files = os.listdir(target_dir)
for file in sample(files, len(files)-1000):
    os.remove(os.path.join(target_dir, file))