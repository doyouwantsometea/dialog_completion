import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from data_loader import DataLoader
from prompter import Prompter
from model_loader import ModelLoader
from utils import extract_json



def arguments():

    parser = ArgumentParser()

    # parser.add_argument('-d', dest='dataset',
    #                     type=str, required=True,
    #                     help='Dataset name to be loaded and processed.')
    
    parser.add_argument('-l', dest='utterance_len',
                        type=int, default=100,
                        help='Minimum token number of utternace to be filled in. (Default=100)')
    
    parser.add_argument('-r', dest='role',
                        type=str, default='Explainer',
                        help='Target speaker role. (Default=\'Explainer\')')

    parser.add_argument('-w', dest='window',
                        type=int, default=2,
                        help='Number of utterances prior to and following the target utterance. (Default=2)')
    
    parser.add_argument('--topic', dest='topic',
                        action='store_true',
                        help='Include topic in the prompt.')
    
    parser.add_argument('--roles', dest='roles',
                        action='store_true',
                        help='Specify speaker roles in the prompt.')
     
    parser.add_argument('--context', dest='context',
                        action='store_true',
                        help='Incorporate dialogue context in prompt footer.')
    
    parser.add_argument('-m', dest='model',
                        type=str, required=True,
                        help='Name of large language model to be loaded via HuggingFace API.')
    
    parser.add_argument('--local', dest='local',
                        action='store_true',
                        help='Load LLM to local device from HuggingFace API.')

    return parser.parse_args()





if __name__ == "__main__":
    
    args = arguments()
    
    df = pd.DataFrame(columns=['file', 'utterance_len', 'role', 'window', 'index',
                               'target_utterance', 'dialogue', 'model', 'topic',
                               'explainer', 'explainee', 'footer_context', 'model_output'])

    prompter = Prompter(prompt_cfg_filename='prompts.json')

    model_loader = ModelLoader(model_name=args.model,
                               local=args.local)



    for root, dirs, files in os.walk('WIRED/data/corpus_dialogs'):
        for file in tqdm(files):
            # print(file)
            # df = pd.read_json(os.path.join(root, file))

            data_loader = DataLoader(path=os.path.join(root, file),
                                     role=args.role,
                                     utterance_len=args.utterance_len,
                                     window=args.window,
                                     replace=True)
            
            index_list = data_loader.filter_utternace()
            topic = data_loader.get_topic()
            explainer, explainee = data_loader.get_dialog_lvl()

            for index in index_list:
                target_utterance, diaolgue = data_loader.parse_diaolgue(index=index)

                # prepare arguments for building prompts
                kwargs = {}
                if args.topic:
                    kwargs['topic'] = topic
                if args.roles:
                    kwargs['explainer'] = explainer
                    kwargs['explainee'] = explainee
                if args.context:
                    kwargs['footer_context'] = True

                prompt = prompter.build_prompt(diaolgue, **kwargs)

                raw_output = model_loader.prompt(prompt).replace(prompt, '')
                print(raw_output)
                json_output = extract_json(raw_output)
                
                if not json_output:
                    continue
                
                model_output = json_output['missing part']
                
                new_row = {
                    'file': file,
                    'utterance_len': args.utterance_len,
                    'role': args.role,
                    'window': args.window,
                    'index': index,
                    'target_utterance': target_utterance,
                    'dialogue': diaolgue,
                    'model': args.model,
                    'topic': topic if args.topic else None,
                    'explainer': explainer if args.roles else None,
                    'explainee': explainee if args.roles else None,
                    'footer_context': True if args.context else False, 
                    'model_output': model_output
                }

                df.loc[len(df)] = new_row
                print(df.head())
    