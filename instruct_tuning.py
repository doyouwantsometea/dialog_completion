import pandas as pd
from argparse import ArgumentParser
from utils import get_worst_features



def arguments():

    parser = ArgumentParser()

    return parser.parse_args()





if __name__ == "__main__":
    
    # args = arguments()

    pass

df = pd.read_json('data/evaluated_results/WIRED/WIRED_Meta-Llama-3.1-8B-Instruct_l30_w2_eval.json')

features = ['interesting', 'engaging', 'specific', 'relevant', 'correct',
            'semantically appropriate', 'understandable', 'fluent',
            'coherent', 'error recovery', 'consistent', 'diverse', 'depth',
            'likeable', 'understand', 'flexible', 'informative', 'inquisitive',
            'minimal_explanations', 'lexical_complexity', 'synonym_density',
            'coherence', 'reading_grade', 'adaptation']

dif_features = []

print(df.columns)

# df.fillna(df.mean(), inplace=True)
# print((df==0).sum())

# df[fed_columns] = df[fed_columns].apply(lambda col: col.fillna(col.mean()))
# df[fed_columns] = df[fed_columns].apply(lambda col: col.replace(0, col.mean()))
# df[ixquisite_columns] = df[ixquisite_columns].apply(lambda col: col.fillna(col.mean()))
# df[ixquisite_columns] = df[ixquisite_columns].apply(lambda col: col.replace(0, col.mean()))

# features = ['interesting', 'lexical_complexity', 'coherence']

for feature in features:
    df[f'{feature}'] = df[f'{feature}'].fillna(df[f'{feature}'].mean())
    df[f'{feature}-original'] = df[f'{feature}-original'].fillna(df[f'{feature}-original'].mean())

    combined_std = pd.concat([df[f'{feature}'], df[f'{feature}-original']]).std()
    # combined_mean = combined_data.mean()
    # combined_std = combined_data.std()

    df[f'{feature}-dif'] = (df[f'{feature}'] - df[f'{feature}-original']) / combined_std
    dif_features.append(f'{feature}-dif')
    # print(df[f'{feature}-dif'].describe())
# print(df[['interesting-dif', 'lexical_complexity-dif', 'coherence-dif']].describe())
# print(df.head())


print(df[dif_features].head())

# print(get_worst_features(df[0:1], 3))

df['worst_features'] = df[dif_features].apply(get_worst_features, n=2, axis=1)


print(df.at[0, 'dialogue'].replace('{missing part}', f'<model-generated> {df.at[0, "model_output"]} </model-generated>'))