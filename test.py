import pandas as pd
import os



for root, dirs, files in os.walk('WIRED/data/corpus_dialogs'):
    for file in files:
        # print(file)

        df = pd.read_json(os.path.join(root, file))
        # print(df.head())


df = pd.read_json('WIRED/data/corpus_dialogs/blackhole_3.json').reset_index(drop=True)
# print(df.reset_index(drop=True))
print(df.columns)
# print(df['turn']) # utterances
# print(df['turn_num_tokens']) # length of speech
print(df[df['role'] == 'Explainer'][df['turn_num_tokens'] > 30].index)



def filter_utternace(df: pd.DataFrame,
                     role: str = 'Explainer',
                     utterance_length: int = 30,
                     window: int = 2):
    # print(len(df))

    parsed_dialogue = str()

    df_filter = df[df.role == role][df.turn_num_tokens > utterance_length]
    for i in df_filter.index:
        start = i - window if window <= i else 0
        end = i + 1 + window if i + 1 + window <= len(df) else len(df)
        print(df[start:end])

        for index, row in df[start:end].iterrows():
            parsed_dialogue += f'{row.role}:'
            if index == i:
                parsed_dialogue += ' {missing part}\n'
            else:
                for sentence in row.turn:
                    parsed_dialogue += f' {sentence}'
                parsed_dialogue += '\n'

    print(parsed_dialogue)

filter_utternace(df)


# text = str()

# for utterance in df[2]['dialog']:
#     text += (utterance['Sentence'][0] + ' ')

# f = open('test.txt', 'a')
# f.write(text)
# f.close()