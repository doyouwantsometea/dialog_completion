import os
import pandas as pd
from argparse import ArgumentParser
from prompter import Prompter
from utils import get_worst_features


def process_feature_stat(df: pd.DataFrame,
                 feature: str):
    # fill empty cells
    df[f'{feature}'] = df[f'{feature}'].fillna(df[f'{feature}'].mean())
    df[f'{feature}-original'] = df[f'{feature}-original'].fillna(df[f'{feature}-original'].mean())
    # normalize with two columns
    combined_std = pd.concat([df[f'{feature}'], df[f'{feature}-original']]).std()
    df[f'{feature}-dif'] = (df[f'{feature}'] - df[f'{feature}-original']) / combined_std


def feature_to_description(worst_features: list,
                           original_prompt: bool = False):
    description = str()
    conj = 'or' if original_prompt else 'and'
    for i, feature in enumerate(worst_features):
        # print(instructions.get(feature[0].split('-')[0]))
        if i+1 == len(worst_features):
            description += f' {conj} {instructions.get(feature[0].split("-")[0])}.'
        else:
            description += f' {instructions.get(feature[0].split("-")[0])},'
    print(description)
    return description


def get_instructions():

    instructions = {
        # FED
        # dialogue-level
        'interesting': 'appear boring',
        'engaging': 'appear unappealing',
        'specific': 'appear out of scope',
        'relevant': 'appear irrelevant',
        'correct': 'misunderstand the conversational context',
        'semantically appropriate': 'make little sense',
        'understandable': 'be hardly understandable',
        'fluent': 'be poorly phrased',
        # turn-level
        'coherent': 'deviate from the topic',
        'error recovery': 'appear errorneous',
        'consistent': 'disagree with previous utterances',
        'diverse': 'include too much repetition',
        'depth': 'appear superficial',
        'likeable': 'appear unfriendly',
        'understand': 'misunderstand the other speaker',
        'flexible': 'adapt poorly to the conversation flow',
        'informative': 'provide too little information',
        'inquisitive': 'appear indifferent',
        # IXQuisite
        'minimal_explanations': 'mention too many named entities',
        'lexical_complexity': 'incorporate difficult word usage',
        'synonym_density': 'paraphrase too little',
        'coherence': 'introduce poor dialogue flow',
        'reading_grade': 'appear too hard to understand',
        'adaptation': 'emphasize the same things too much'
    }

    return instructions



def arguments():

    parser = ArgumentParser()

    parser.add_argument('-n', dest='num_feature',
                        type=int, default=3,
                        help='Number of features to be tuned with instructions.')
    
    parser.add_argument('--original_prompt', dest='original_prompt',
                        action='store_true',
                        help='Use the same prompt for running the task (plus instructions), otherwise adopt a different prompt structure.')

    return parser.parse_args()





if __name__ == "__main__":
    
    args = arguments()


    # for root, dirs, files in os.walk('data/evaluated_results'):
    #     for file in files:
            
    #         df = pd.read_json(os.path.join(root, file))


    
    df = pd.read_json('data/evaluated_results/WIRED/WIRED_Meta-Llama-3.1-8B-Instruct_l30_w2_eval.json')


    instructions = get_instructions()
    features = list(instructions.keys())

    dif_features = [f'{feature}-dif' for feature in features]

    print(df.columns)

    # df.fillna(df.mean(), inplace=True)
    # print((df==0).sum())

    # df[fed_columns] = df[fed_columns].apply(lambda col: col.fillna(col.mean()))
    # df[fed_columns] = df[fed_columns].apply(lambda col: col.replace(0, col.mean()))
    # df[ixquisite_columns] = df[ixquisite_columns].apply(lambda col: col.fillna(col.mean()))
    # df[ixquisite_columns] = df[ixquisite_columns].apply(lambda col: col.replace(0, col.mean()))

    # features = ['interesting', 'lexical_complexity', 'coherence']

    for feature in features:
        process_feature_stat(df, feature)
        # df[f'{feature}'] = df[f'{feature}'].fillna(df[f'{feature}'].mean())
        # df[f'{feature}-original'] = df[f'{feature}-original'].fillna(df[f'{feature}-original'].mean())

        # combined_std = pd.concat([df[f'{feature}'], df[f'{feature}-original']]).std()
        # # combined_mean = combined_data.mean()
        # # combined_std = combined_data.std()

        # df[f'{feature}-dif'] = (df[f'{feature}'] - df[f'{feature}-original']) / combined_std
        # dif_features.append(f'{feature}-dif')
        # print(df[f'{feature}-dif'].describe())
    # print(df[['interesting-dif', 'lexical_complexity-dif', 'coherence-dif']].describe())
    # print(df.head())


    print(df[dif_features].head())

    # print(get_worst_features(df[0:1], 3))

    df['worst_features'] = df[dif_features].apply(get_worst_features,
                                                  n=args.num_feature,
                                                  axis=1)


    # print(df.at[0, 'dialogue'].replace('{missing part}', f'<model-generated> {df.at[0, "model_output"]} </model-generated>'))


    prompter = Prompter(prompt_cfg_filename='prompts.json',
                        task='task' if args.original_prompt else 'tuning')
    
    for index, row in df.iterrows():
        
        if args.original_prompt:
            dialogue = row.dialogue
        else:
            dialogue = row.dialogue.replace('{missing part}', f'<model-generated> {row.model_output} </model-generated>')

        print(row.worst_features)
        description = feature_to_description(worst_features=row.worst_features,
                                             original_prompt=args.original_prompt)
        prompt = prompter.build_prompt(dialogue=dialogue,
                                       instruction=description)
        print(prompt)