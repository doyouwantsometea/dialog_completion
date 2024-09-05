import fed
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from IXQuisite import IXQuisite
from utils import flatten_dialogue

# path for downloading LLMs
os.makedirs('llm_cache', exist_ok=True)
os.environ['HF_HOME'] = 'llm_cache'

def arguments():

    parser = ArgumentParser()

    parser.add_argument('-d', dest='dataset',
                        type=str, required=True,
                        help='Dataset to be evaluated. Currently available options: WIRED, WikiDialog, ELI5.')
    
    parser.add_argument('--fed', dest='FED',
                        action='store_true',
                        help='Minimum token number of utternace to be filled in. (Default=100)')
    
    parser.add_argument('--ixquisite', dest='IXQuisite',
                        action='store_true',
                        help='Target speaker role. (Default=\'Explainer\')')
    
    parser.add_argument('--tuned', dest='tuned',
                        default=False, action='store_true',
                        help='Evaluate fine-tuned model output instead of task results.')

    return parser.parse_args()



if __name__ == "__main__":
    
    args = arguments()

    if args.FED:
        model, tokenizer = fed.load_models('microsoft/DialoGPT-large')

    path = f'data/tuned_results/{args.dataset}' if args.tuned else f'data/results/{args.dataset}'
    for root, dirs, files in os.walk(path):
        for file in files:
            
            df = pd.read_json(os.path.join(root, file))
            
            new_col = []
            if args.FED:
                if args.tuned:
                    new_col.extend(['interesting-tuned', 'engaging-tuned',
                        'specific-tuned', 'relevant-tuned', 'correct-tuned',
                        'semantically appropriate-tuned', 'understandable-tuned',
                        'fluent-tuned', 'coherent-tuned', 'error recovery-tuned',
                        'consistent-tuned', 'diverse-tuned', 'depth-tuned',
                        'likeable-tuned', 'understand-tuned', 'flexible-tuned',
                        'informative-tuned', 'inquisitive-tuned'])
                else:
                    new_col.extend(['interesting', 'engaging', 'specific', 'relevant',
                        'correct', 'semantically appropriate', 'understandable', 'fluent',
                        'coherent', 'error recovery', 'consistent', 'diverse', 'depth',
                        'likeable', 'understand', 'flexible', 'informative', 'inquisitive'])
                    new_col.extend(['interesting-original', 'engaging-original',
                        'specific-original','relevant-original', 'correct-original',
                        'semantically appropriate-original', 'understandable-original',
                        'fluent-original', 'coherent-original', 'error recovery-original',
                        'consistent-original', 'diverse-original', 'depth-original',
                        'likeable-original', 'understand-original', 'flexible-original',
                        'informative-original', 'inquisitive-original'])
            if args.IXQuisite:
                if args.tuned:
                    new_col.extend(['minimal_explanations-tuned', 'lexical_complexity-tuned',
                        'synonym_density-tuned', 'coherence-tuned', 'reading_grade-tuned',
                        'adaptation-tuned'])
                else:
                    new_col.extend(['minimal_explanations', 'lexical_complexity',
                        'synonym_density', 'coherence', 'reading_grade', 'adaptation',
                        'minimal_explanations-original', 'lexical_complexity-original',
                        'synonym_density-original', 'coherence-original',
                        'reading_grade-original', 'adaptation-original'])
                
            df[new_col] = None


            for index, row in tqdm(df.iterrows(),
                                   total=df.shape[0],
                                   desc=f'Processing DF'):
                print(row.model_output)
                if not isinstance(row.model_output, str):
                    print('Skipping instance owing to empty or improper model output.')
                    continue

                if args.tuned and not isinstance(row.tuned_output, str):
                    print('Skipping instance owing to empty or improper tuned output.')
                    continue

                if args.FED:
                    conversation = flatten_dialogue(dialogue=row.dialogue,
                                                    reference=row.target_turn,
                                                    model_turn=row.tuned_output if args.tuned else row.model_output,
                                                    original_dialog=False)

                    scores = fed.evaluate(conversation, model, tokenizer)
                    print(scores)

                    
                    if args.tuned:
                        df.at[index, 'interesting-tuned'] = round(scores['interesting'] * 100, 4)
                        df.at[index, 'engaging-tuned'] = round(scores['engaging'] * 100, 4)
                        df.at[index, 'specific-tuned'] = round(scores['specific'] * 100, 4)
                        df.at[index, 'relevant-tuned'] = round(scores['relevant'], 4)
                        df.at[index, 'correct-tuned'] = round(scores['correct'], 4)
                        df.at[index, 'semantically appropriate-tuned'] = round(scores['semantically appropriate'] * 100, 4)
                        df.at[index, 'understandable-tuned'] = round(scores['understandable'] * 100, 4)
                        df.at[index, 'fluent-tuned'] = round(scores['fluent'] * 100, 4)
                        df.at[index, 'coherent-tuned'] = round(scores['coherent'], 4)
                        df.at[index, 'error recovery-tuned'] = round(scores['error recovery'], 4)
                        df.at[index, 'consistent-tuned'] = round(scores['consistent'], 4)
                        df.at[index, 'diverse-tuned'] = round(scores['diverse'], 4)
                        df.at[index, 'depth-tuned'] = round(scores['depth'], 4)
                        df.at[index, 'likeable-tuned'] = round(scores['likeable'] * 100, 4)
                        df.at[index, 'understand-tuned'] = round(scores['understand'], 4)
                        df.at[index, 'flexible-tuned'] = round(scores['flexible'] * 100, 4)
                        df.at[index, 'informative-tuned'] = round(scores['informative'] * 100, 4)
                        df.at[index, 'inquisitive-tuned'] = round(scores['inquisitive'] * 100, 4)
                    
                    else:
                        conversation_original = flatten_dialogue(dialogue=row.dialogue,
                                                                 reference=row.target_turn,
                                                                 model_turn=row.model_output,
                                                                 original_dialog=True)

                        scores_original = fed.evaluate(conversation_original, model, tokenizer)
                        print(scores_original)

                        df.at[index, 'interesting'] = round(scores['interesting'] * 100, 4)
                        df.at[index, 'engaging'] = round(scores['engaging'] * 100, 4)
                        df.at[index, 'specific'] = round(scores['specific'] * 100, 4)
                        df.at[index, 'relevant'] = round(scores['relevant'], 4)
                        df.at[index, 'correct'] = round(scores['correct'], 4)
                        df.at[index, 'semantically appropriate'] = round(scores['semantically appropriate'] * 100, 4)
                        df.at[index, 'understandable'] = round(scores['understandable'] * 100, 4)
                        df.at[index, 'fluent'] = round(scores['fluent'] * 100, 4)
                        df.at[index, 'coherent'] = round(scores['coherent'], 4)
                        df.at[index, 'error recovery'] = round(scores['error recovery'], 4)
                        df.at[index, 'consistent'] = round(scores['consistent'], 4)
                        df.at[index, 'diverse'] = round(scores['diverse'], 4)
                        df.at[index, 'depth'] = round(scores['depth'], 4)
                        df.at[index, 'likeable'] = round(scores['likeable'] * 100, 4)
                        df.at[index, 'understand'] = round(scores['understand'], 4)
                        df.at[index, 'flexible'] = round(scores['flexible'] * 100, 4)
                        df.at[index, 'informative'] = round(scores['informative'] * 100, 4)
                        df.at[index, 'inquisitive'] = round(scores['inquisitive'] * 100, 4)

                        df.at[index, 'interesting-original'] = round(scores_original['interesting'] * 100, 4)
                        df.at[index, 'engaging-original'] = round(scores_original['engaging'] * 100, 4)
                        df.at[index, 'specific-original'] = round(scores_original['specific'] * 100, 4)
                        df.at[index, 'relevant-original'] = round(scores_original['relevant'], 4)
                        df.at[index, 'correct-original'] = round(scores_original['correct'], 4)
                        df.at[index, 'semantically appropriate-original'] = round(scores_original['semantically appropriate'] * 100, 4)
                        df.at[index, 'understandable-original'] = round(scores_original['understandable'] * 100, 4)
                        df.at[index, 'fluent-original'] = round(scores_original['fluent'] * 100, 4)
                        df.at[index, 'coherent-original'] = round(scores_original['coherent'], 4)
                        df.at[index, 'error recovery-original'] = round(scores_original['error recovery'], 4)
                        df.at[index, 'consistent-original'] = round(scores_original['consistent'], 4)
                        df.at[index, 'diverse-original'] = round(scores_original['diverse'], 4)
                        df.at[index, 'depth-original'] = round(scores_original['depth'], 4)
                        df.at[index, 'likeable-original'] = round(scores_original['likeable'] * 100, 4)
                        df.at[index, 'understand-original'] = round(scores_original['understand'], 4)
                        df.at[index, 'flexible-original'] = round(scores_original['flexible'] * 100, 4)
                        df.at[index, 'informative-original'] = round(scores_original['informative'] * 100, 4)
                        df.at[index, 'inquisitive-original'] = round(scores_original['inquisitive'] * 100, 4)



                if args.IXQuisite:
                    ts = IXQuisite(datapoint=row.to_dict(),
                                   original_dialog=False,
                                   tuned=args.tuned,
                                   r=4)
                    scores = ts.get_scores()
                    print(scores)

                    
                    if args.tuned:
                        df.at[index, 'minimal_explanations-tuned'] = scores['minimal_explanations']
                        df.at[index, 'lexical_complexity-tuned'] = scores['lexical_complexity']
                        df.at[index, 'synonym_density-tuned'] = scores['synonym_density']
                        df.at[index, 'coherence-tuned'] = scores['coherence']
                        df.at[index, 'reading_grade-tuned'] = scores['reading_grade']
                        df.at[index, 'adaptation-tuned'] = scores['adaptation']
                    
                    else: 
                        ts_original = IXQuisite(datapoint=row.to_dict(),
                                                original_dialog=True,
                                                r=4)
                        
                        scores_original = ts_original.get_scores()
                        print(scores_original)
                        
                        df.at[index, 'minimal_explanations'] = scores['minimal_explanations']
                        df.at[index, 'lexical_complexity'] = scores['lexical_complexity']
                        df.at[index, 'synonym_density'] = scores['synonym_density']
                        df.at[index, 'coherence'] = scores['coherence']
                        df.at[index, 'reading_grade'] = scores['reading_grade']
                        df.at[index, 'adaptation'] = scores['adaptation']

                        df.at[index, 'minimal_explanations-original'] = scores_original['minimal_explanations']
                        df.at[index, 'lexical_complexity-original'] = scores_original['lexical_complexity']
                        df.at[index, 'synonym_density-original'] = scores_original['synonym_density']
                        df.at[index, 'coherence-original'] = scores_original['coherence']
                        df.at[index, 'reading_grade-original'] = scores_original['reading_grade']
                        df.at[index, 'adaptation-original'] = scores_original['adaptation']

                    print(df.head()) 
            
            os.makedirs(f'data/evaluated_results/{args.dataset}', exist_ok=True)

            df.to_json(f'data/evaluated_results/{file.split(".json")[0]}_eval.json')