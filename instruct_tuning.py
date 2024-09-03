import os
import pandas as pd
from argparse import ArgumentParser
from prompter import Prompter
from model_loader import HFModelLoader, AnthropicModelLoader
from utils import get_worst_features, extract_json


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
    
    desc_neg = str()
    desc_pos = str()

    conj = 'or' if original_prompt else 'and'
    for i, feature in enumerate(worst_features):
        # print(instructions.get(feature[0].split('-')[0]))
        if i+1 == len(worst_features):
            desc_neg += f' {conj} {instructions.get(feature[0].split("-")[0])[0]},'
            desc_pos += f' and {instructions.get(feature[0].split("-")[0])[1]}.'
        else:
            desc_neg += f' {instructions.get(feature[0].split("-")[0])[0]},'
            desc_pos += f' {instructions.get(feature[0].split("-")[0])[1]},'
    
    description = f'{desc_neg} so that the conversation becomes more{desc_pos}'
    return description


def get_instructions():

    instructions = {
        # FED
        # dialogue-level
        'interesting': ('appear boring', 'interesting'),
        'engaging': ('appear unappealing', 'engaging'),
        'specific': ('appear out of scope', 'topic-specific'),
        'relevant': ('appear irrelevant', 'topic-relevant'),
        'correct': ('misunderstand the conversational context', 'correct'),
        'semantically appropriate': ('make little sense', 'semantically appropriate'),
        'understandable': ('be hardly understandable', 'understandable'),
        'fluent': ('be poorly phrased', 'fluent'),
        # turn-level
        'coherent': ('deviate from the topic', 'coherent'),
        'error recovery': ('appear erroneous', 'self-corrective'),
        'consistent': ('disagree with previous utterances', 'consistent'),
        'diverse': ('include too much repetition', 'lexically diverse'),
        'depth': ('appear superficial', 'in-depth'),
        'likeable': ('appear unfriendly', 'likeable'),
        'understand': ('misunderstand the other speaker', 'perceptive'),
        'flexible': ('adapt poorly to the conversation flow', 'flexible'),
        'informative': ('provide too little information', 'informative'),
        'inquisitive': ('appear indifferent', 'inquisitive'),
        # IXQuisite
        'minimal_explanations': ('mention too many named entities', 'accessible'),
        'lexical_complexity': ('incorporate difficult word usage', 'colloquial'),
        'synonym_density': ('paraphrase too little', 'lexically diverse'),
        'coherence': ('introduce poor dialogue flow', 'coherent'),
        'reading_grade': ('appear too hard to understand', 'plain'),
        'adaptation': ('emphasize the same things too much', 'adaptive')
    }

    return instructions



def arguments():

    parser = ArgumentParser()

    parser.add_argument('-n', dest='num_feature',
                        type=int, default=3,
                        help='Number of features to be tuned with instructions.')
    
    parser.add_argument('-m', dest='model',
                        type=str, required=True,
                        help='Large Language Model for tuning output.')

    parser.add_argument('--local', dest='local',
                        action='store_true',
                        help='Download LLM to local device from HuggingFace. (Not applicable to Anthropic models.)')
    
    parser.add_argument('--original_prompt', dest='original_prompt',
                        action='store_true',
                        help='Use the same prompt for running the task (plus instructions), otherwise adopt a different prompt structure.')

    return parser.parse_args()





if __name__ == "__main__":
    
    args = arguments()

    if 'claude' in args.model:
        model_loader = AnthropicModelLoader(model_name=args.model)
    else:
        model_loader = HFModelLoader(model_name=args.model,
                                     local=args.local)

    # initiate instructions and features
    instructions = get_instructions()
    features = list(instructions.keys())
    dif_features = [f'{feature}-dif' for feature in features]

    for root, dirs, files in os.walk('data/evaluated_results'):
        # filter files with the model's output
        for file in (f for f in files if args.model == f.split('_')[1]):
            print(file)
            # if args.model != file.split('_')[1]:
            #     continue
            
            df = pd.read_json(os.path.join(root, file))

    

    
            # df = pd.read_json('data/evaluated_results/WIRED/WIRED_Meta-Llama-3.1-8B-Instruct_l30_w2_eval.json')

            print(df.columns)

            # process features
            for feature in features:
                process_feature_stat(df, feature)

            print(df[dif_features].head())

            # print(get_worst_features(df[0:1], 3))

            # assign new columns
            df['worst_features'] = df[dif_features].apply(get_worst_features,
                                                        n=args.num_feature,
                                                        axis=1)
            df['tuned_output'] = None

            # print(df.at[0, 'dialogue'].replace('{missing part}', f'<model-generated> {df.at[0, "model_output"]} </model-generated>'))



            prompter = Prompter(prompt_cfg_filename='prompts.json',
                                task='task' if args.original_prompt else 'tuning')
            
            for index, row in df.iterrows():
                
                if args.original_prompt:
                    dialogue = row.dialogue
                else:
                    dialogue = row.dialogue.replace('{missing part}', f'<model-generated> {row.model_output} </model-generated>')

                # print(row.worst_features)
                description = feature_to_description(worst_features=row.worst_features,
                                                    original_prompt=args.original_prompt)
                prompt = prompter.build_prompt(dialogue=dialogue,
                                            instruction=description)
                print(prompt)

                raw_output = model_loader.prompt(prompt).replace(prompt, '')
                print(raw_output)
                json_output = extract_json(raw_output)
                # print(json_output)
                if not json_output:
                    continue
                        
                model_output = json_output.get('revised turn', None)
                print(model_output)
                df.at[index, 'tuned_output'] = model_output
                print(df.head())
        
        os.makedirs('data/tuned_results', exist_ok=True)
        df.to_json(f'data/tuned_results/{file.split(".json")[0]}_tuned.json')