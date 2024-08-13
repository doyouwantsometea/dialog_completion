import json
import os
import pandas as pd


for root, dirs, files in os.walk('data/WIRED/data/json'):
    # print(file)

    for file in files:
        print(file)
        with open(f'{root}/{file}', 'r') as f:
            data = json.load(f)[0]['data']

        # print(data)
        dialog = data['dialogue']

        data['topic']
        data['level']
        data['youtube_link']

        # df = pd.read_json('data/WIRED/data/corpus_dialogs/blackhole_3.json')
        # print(df.columns)

        df = pd.DataFrame(columns=['topic', 'youtube_link', 'wired_link',
                                'dialog_lvl', 'dialog_id', 'speaker_id',
                                'reciever_id', 'turn_id', 'turn', 'turn_sentences',
                                'turn_num_tokens', 'role'])

        for turn in dialog:
            df.at[len(df), 'turn_id'] = len(df)
            df.at[len(df)-1, 'role'] = turn['author']
            df.at[len(df)-1, 'turn'] = [turn['text']]
            df.at[len(df)-1, 'turn_sentences'] = [turn['text']]
            df.at[len(df)-1, 'turn_num_tokens'] = len(turn['text'].split(' '))
            df.at[len(df)-1, 'topic'] = data['topic']
            df.at[len(df)-1, 'dialog_lvl'] = data['level']
            df.at[len(df)-1, 'wired_link'] = data['youtube_link']

        print(df.head())
        file_name = file.split('_processed.json')[0]

        df.to_json(f'data/WIRED/data/corpus_dialogs/{file_name}.json')