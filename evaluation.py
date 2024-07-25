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
                    conversation = str()
                    conversation = flatten_dialogue(dialogue=row.dialogue,
                                                    reference=row.target_utterance,
                                                    utterance=row.model_output,
                                                    original_dialog=False)

                    scores = fed.evaluate(conversation, model, tokenizer)
                
                
                if args.IXQuisite:
                    ts = IXQuisite(datapoint=row.to_dict(),
                                   original_dialog=False,
                                   r=4)