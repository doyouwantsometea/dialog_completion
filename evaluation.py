import fed
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from IXQuisite import IXQuisite
from utils import flatten_dialogue

# path for downloading LLMs
os.makedirs('llm_cache', exist_ok=True)
os.environ['HF_HOME'] = './llm_cache'

def arguments():

    parser = ArgumentParser()

    parser.add_argument('--fed', dest='FED',
                        action='store_true',
                        help='Minimum token number of utternace to be filled in. (Default=100)')
    
    parser.add_argument('--ixquisite', dest='IXQuisite',
                        action='store_true',
                        help='Target speaker role. (Default=\'Explainer\')')

    return parser.parse_args()



if __name__ == "__main__":
    
    args = arguments()

    if args.FED:
        model, tokenizer = fed.load_models('microsoft/DialoGPT-large')


    for root, dirs, files in os.walk('data/results'):
        for file in files:
            
            df = pd.read_json(os.path.join(root, file))
            
            new_col = []
            if args.FED:
                new_col.extend(['interesting', 'engaging', 'specific', 'relevant',
                    'correct', 'semantically appropriate', 'understandable', 'fluent',
                    'coherent', 'error recovery', 'consistent', 'diverse', 'depth',
                    'likeable', 'understand', 'flexible', 'informative', 'inquisitive',
                    'interesting-original', 'engaging-original', 'specific-original',
                    'relevant-original', 'correct-original',
                    'semantically appropriate-original', 'understandable-original',
                    'fluent-original', 'coherent-original', 'error recovery-original',
                    'consistent-original', 'diverse-original', 'depth-original',
                    'likeable-original', 'understand-original', 'flexible-original',
                    'informative-original', 'inquisitive-original'])
            if args.IXQuisite:
                new_col.extend(['minimal_explanations', 'lexical_complexity',
                    'synonym_density', 'coherence', 'reading_grade', 'adaptation',
                    'minimal_explanations-original', 'lexical_complexity-original',
                    'synonym_density-original', 'coherence-original',
                    'reading_grade-original', 'adaptation-original'])
                
            df[new_col] = None


            for index, row in tqdm(df.iterrows(),
                                   total=df.shape[0],
                                   desc=f'Processing DF'):

                if not row.model_output or len(str(row.model_output).split(' ')) < 10:
                    print('Skipping instance owing to empty or short model output.')
                    continue

                if args.FED:
                    conversation = flatten_dialogue(dialogue=row.dialogue,
                                                    reference=row.target_utterance,
                                                    utterance=row.model_output,
                                                    original_dialog=False)

                    scores = fed.evaluate(conversation, model, tokenizer)

                    conversation_original = flatten_dialogue(dialogue=row.dialogue,
                                                             reference=row.target_utterance,
                                                             utterance=row.model_output,
                                                             original_dialog=True)

                    scores_original = fed.evaluate(conversation_original, model, tokenizer)

                    row['interesting'] = round(scores['interesting'] * 100, 4)
                    row['engaging'] = round(scores['engaging'] * 100, 4)
                    row['specific'] = round(scores['specific'] * 100, 4)
                    row['relevant'] = round(scores['relevant'], 4)
                    row['correct'] = round(scores['correct'], 4)
                    row['semantically appropriate'] = round(scores['semantically appropriate'] * 100, 4)
                    row['understandable'] = round(scores['understandable'] * 100, 4)
                    row['fluent'] = round(scores['fluent'] * 100, 4)
                    row['coherent'] = round(scores['coherent'], 4)
                    row['error recovery'] = round(scores['error recovery'], 4)
                    row['consistent'] = round(scores['consistent'], 4)
                    row['diverse'] = round(scores['diverse'], 4)
                    row['depth'] = round(scores['depth'], 4)
                    row['likeable'] = round(scores['likeable'] * 100, 4)
                    row['understand'] = round(scores['understand'], 4)
                    row['flexible'] = round(scores['flexible'] * 100, 4)
                    row['informative'] = round(scores['informative'] * 100, 4)
                    row['inquisitive'] = round(scores['inquisitive'] * 100, 4)

                    row['interesting-original'] = round(scores_original['interesting'] * 100, 4)
                    row['engaging-original'] = round(scores_original['engaging'] * 100, 4)
                    row['specific-original'] = round(scores_original['specific'] * 100, 4)
                    row['relevant-original'] = round(scores_original['relevant'], 4)
                    row['correct-original'] = round(scores_original['correct'], 4)
                    row['semantically appropriate-original'] = round(scores_original['semantically appropriate'] * 100, 4)
                    row['understandable-original'] = round(scores_original['understandable'] * 100, 4)
                    row['fluent-original'] = round(scores_original['fluent'] * 100, 4)
                    row['coherent-original'] = round(scores_original['coherent'], 4)
                    row['error recovery-original'] = round(scores_original['error recovery'], 4)
                    row['consistent-original'] = round(scores_original['consistent'], 4)
                    row['diverse-original'] = round(scores_original['diverse'], 4)
                    row['depth-original'] = round(scores_original['depth'], 4)
                    row['likeable-original'] = round(scores_original['likeable'] * 100, 4)
                    row['understand-original'] = round(scores_original['understand'], 4)
                    row['flexible-original'] = round(scores_original['flexible'] * 100, 4)
                    row['informative-original'] = round(scores_original['informative'] * 100, 4)
                    row['inquisitive-original'] = round(scores_original['inquisitive'] * 100, 4)



                if args.IXQuisite:
                    ts = IXQuisite(datapoint=row.to_dict(),
                                   original_dialog=False,
                                   r=4)
                    scores = ts.get_scores()

                    ts_original = IXQuisite(datapoint=row.to_dict(),
                                            original_dialog=True,
                                            r=4)
                    
                    scores_original = ts.get_scores()


                    row['minimal_explanations'] = scores['minimal_explanations']
                    row['lexical_complexity'] = scores['lexical_complexity']
                    row['synonym_density'] = scores['synonym_density']
                    row['coherence'] = scores['coherence']
                    row['reading_grade'] = scores['reading_grade']
                    row['adaptation'] = scores['adaptation']

                    row['minimal_explanations-original'] = scores_original['minimal_explanations']
                    row['lexical_complexity-original'] = scores_original['lexical_complexity']
                    row['synonym_density-original'] = scores_original['synonym_density']
                    row['coherence-original'] = scores_original['coherence']
                    row['reading_grade-original'] = scores_original['reading_grade']
                    row['adaptation-original'] = scores_original['adaptation']

            
            os.makedirs('data/evaluated_results', exist_ok=True)

            df.to_json(f'data/evaluated_results/{file}_eval.json')