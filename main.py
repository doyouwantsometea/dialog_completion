import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from data_loader import DataLoader
from prompter import Prompter
from model_loader import ModelLoader
from utils import extract_json, remove_training_set



def arguments():

    parser = ArgumentParser()

    # parser.add_argument('-d', dest='dataset',
    #                     type=str, required=True,
    #                     help='Dataset name to be loaded and processed.')
    
    parser.add_argument('-l', dest='turn_len',
                        type=int, default=100,
                        help='Minimum token number of utternace to be filled in. (Default=100)')
    
    parser.add_argument('-r', dest='role',
                        type=str, default='Explainer',
                        help='Target speaker role. (Default=\'Explainer\')')

    parser.add_argument('-w', dest='window',
                        type=int, default=2,
                        help='Number of turns prior to and following the target turn. (Default=2)')
    
    parser.add_argument('--topic', dest='topic',
                        action='store_true',
                        help='Include topic in the prompt.')
    
    parser.add_argument('--speakers', dest='speakers',
                        action='store_true',
                        help='Specify speaker roles in the prompt.')
     
    parser.add_argument('--context', dest='context',
                        action='store_true',
                        help='Incorporate dialogue context in prompt footer.')
    
    parser.add_argument('--open_end', dest='open_end',
                        action='store_true',
                        help='Remove the tunrs occuring after the target turn.')
    
    parser.add_argument('-d', dest='dataset',
                        type=str, required=True,
                        help='Dataset. Currently available options: WIRED, WikiDialog.')

    parser.add_argument('-m', dest='model',
                        type=str, required=True,
                        help='Name of large language model to be loaded via HuggingFace API.')
    
    parser.add_argument('--local', dest='local',
                        action='store_true',
                        help='Download LLM to local device from HuggingFace.')

    return parser.parse_args()





if __name__ == "__main__":
    
    args = arguments()
    
    df = pd.DataFrame(columns=['file', 'turn_len', 'role', 'window', 'index',
                               'target_turn', 'dialogue', 'model', 'topic',
                               'explainer', 'explainee', 'footer_context', 'model_output'])

    prompter = Prompter(prompt_cfg_filename='prompts.json')

    model_loader = ModelLoader(model_name=args.model,
                               local=args.local)


    if args.dataset == 'WIRED':
        path = 'data/WIRED/data/corpus_dialogs'
    elif args.dataset == 'WikiDialog':
        path = 'data/WikiDialog'
    else:
        raise ValueError('Invalid dataset.')
    
    print(os.walk(path))


    for root, dirs, files in os.walk(path):
        # print(len(files))
        files = remove_training_set(files)
        # print(files)

        for file in tqdm(files):

            if file == '.DS_Store':
                continue
            # print(file)
            # df = pd.read_json(os.path.join(root, file))

            data_loader = DataLoader(dataset=args.dataset,
                                     path=os.path.join(root, file),
                                     role=args.role,
                                     turn_len=args.turn_len,
                                     window=args.window,
                                     replace=True)
            
            index_list = data_loader.filter_turn()
            topic = data_loader.get_topic()
            explainer, explainee = data_loader.get_dialog_lvl()

            for index in index_list:
                target_turn, diaolgue = data_loader.parse_diaolgue(index=index,
                                                                   open_end=args.open_end)

                # prepare arguments for building prompts
                kwargs = {}
                if args.topic:
                    kwargs['topic'] = topic
                if args.speakers:
                    kwargs['explainer'] = explainer
                    kwargs['explainee'] = explainee
                if args.context:
                    kwargs['footer_context'] = True

                prompt = prompter.build_prompt(diaolgue, **kwargs)
                
                print(prompt)
                
                raw_output = model_loader.prompt(prompt).replace(prompt, '')
                print(raw_output)
                json_output = extract_json(raw_output)
                
                if not json_output:
                    continue
                
                model_output = json_output.get('missing part', None)
                
                new_row = {
                    'file': file,
                    'turn_len': args.turn_len,
                    'role': args.role,
                    'window': args.window,
                    'index': index,
                    'target_turn': target_turn,
                    'dialogue': diaolgue,
                    'model': args.model,
                    'topic': topic if args.topic else None,
                    'explainer': explainer if args.speakers else None,
                    'explainee': explainee if args.speakers else None,
                    'footer_context': True if args.context else False, 
                    'model_output': model_output
                }

                df.loc[len(df)] = new_row
                print(df.head())
        
    os.makedirs('data/results', exist_ok=True)

    optional_args = [f'{"topic" if args.topic else ""}',
                     f'{"speakers" if args.speakers else ""}',
                     f'{"context" if args.context else ""}',
                     f'{"openend" if args.open_end else ""}']
    file_name_suffix = str()
    for arg in optional_args:
        if arg != '':
            file_name_suffix += f'_{arg}'

    file_name = f'{args.dataset}_{args.model}_l{args.turn_len}_w{args.window}'
    df.to_json(f'data/results/{file_name}{file_name_suffix}.json')