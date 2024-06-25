import pandas as pd
import os
from data_loader import DataLoader


for root, dirs, files in os.walk('WIRED/data/corpus_dialogs'):
    for file in files:
        # print(file)

        df = pd.read_json(os.path.join(root, file))
        # print(df.head())


# df = pd.read_json('WIRED/data/corpus_dialogs/blackhole_3.json').reset_index(drop=True)
# print(df.reset_index(drop=True))
# print(df.columns)
# print(df['turn']) # utterances
# print(df['turn_num_tokens']) # length of speech
# print(df[df['role'] == 'Explainer'][df['turn_num_tokens'] > 30].index)



# def filter_utternace(df: pd.DataFrame,
#                      role: str = 'Explainer',
#                      utterance_len: int = 30):
#     """
#     Filter utterances with the given conditions.
#     :param df: Dataframe of loaded WIRED data.
#     :param role: Speaker of the desired utterances (Explainer / Explainee).
#     :param utterance_len: Minimum token numbers of the desired utterances.
#     :return: List of indexes that point toward utterances in the dataframe.
#     """
#     df_filter = df[(df.role == role) & (df.turn_num_tokens > utterance_len)]
#     return df_filter.index


# def parse_diaolgue(df: pd.DataFrame,
#                    index: str,
#                    window: str = 2,
#                    replace: bool = True):
#     """
#     Parse utterances into dialogue segment for prompting.
#     :param df: Dataframe of loaded WIRED data.
#     :param index: Index of the filtered utterance.
#     :param window: Number of prior and following utterances to be included in the dialogue segment.
#     :param replace: Whether to replace the filtered utterance with {missing part} placeholder.
#     :return: Dialogue string that can be prompted.
#     """
#     start = index - window if window <= index else 0
#     end = index + 1 + window if index + 1 + window <= len(df) else len(df)
#     print(df[start:end])

#     parsed_dialogue = str()
    
#     for i, r in df[start:end].iterrows():
#         parsed_dialogue += f'{r.role}:'
#         if i == index and replace:
#             parsed_dialogue += ' {missing part}\n'
#         else:
#             for sentence in r.turn:
#                 parsed_dialogue += f' {sentence}'
#             parsed_dialogue += '\n'

#     return parsed_dialogue

data_loader = DataLoader(path = 'WIRED/data/corpus_dialogs/blackhole_3.json',
                         role = 'Explainer',
                         utterance_len = 30,
                         window = 2,
                         replace = True)

index_list = DataLoader.filter_utternace(data_loader)
for index in index_list:
    parsed_diaolgue = DataLoader.parse_diaolgue(data_loader, index=index)
    print(parsed_diaolgue)
# text = str()

# for utterance in df[2]['dialog']:
#     text += (utterance['Sentence'][0] + ' ')

# f = open('test.txt', 'a')
# f.write(text)
# f.close()