import fed
import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from IXQuisite import IXQuisite
from utils import flatten_dialogue


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
            
            
            for index, row in tqdm(df.iterrows(),
                                   total=df.shape[0],
                                   desc=f'Processing DF'):
                
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

                    scores['interesting'] * 100
                    scores['engaging'] * 100
                    scores['specific'] * 100
                    scores['relevant']
                    scores['correct']
                    scores['semantically appropriate'] * 100
                    scores['understandable'] * 100
                    scores['fluent'] * 100
                    scores['coherent']
                    scores['error recovery']
                    scores['consistent']
                    scores['diverse']
                    scores['depth']
                    scores['likeable'] * 100
                    scores['understand']
                    scores['flexible'] * 100
                    scores['informative'] * 100
                    scores['inquisitive'] * 100



                if args.IXQuisite:
                    ts = IXQuisite(datapoint=row.to_dict(),
                                   original_dialog=False,
                                   r=4)
                    scores = ts.get_scores()

                    scores['minimal_explanations']
                    scores['lexical_complexity']
                    scores['synonym_density']
                    scores['coherence']
                    scores['reading_grade']
                    scores['adaptation']

                    ts_original = IXQuisite(datapoint=row.to_dict(),
                                            original_dialog=True,
                                            r=4)