import pandas as pd
import os
from data_loader import DataLoader
from prompter import Prompter


for root, dirs, files in os.walk('WIRED/data/corpus_dialogs'):
    for file in files:
        # print(file)

        df = pd.read_json(os.path.join(root, file))
        # print(df.head())


data_loader = DataLoader(path='WIRED/data/corpus_dialogs/blackhole_3.json',
                         role='Explainer',
                         utterance_len=30,
                         window=2,
                         replace=True)

prompter = Prompter(prompt_cfg_filename='prompts.json')



index_list = data_loader.filter_utternace()
for index in index_list:
    diaolgue = data_loader.parse_diaolgue(index=index)
    # print(diaolgue)
    prompter.build_prompt(diaolgue)
# text = str()

# for utterance in df[2]['dialog']:
#     text += (utterance['Sentence'][0] + ' ')

# f = open('test.txt', 'a')
# f.write(text)
# f.close()