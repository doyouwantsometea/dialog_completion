import os
import json
import pandas as pd
from argparse import ArgumentParser


def parse_json(df):

    dialogue = []
    for index, row in df.iterrows():
        author = row.role
        text = str()
        for sentence in row.turn:
            text += f'{sentence} '
        text = text.rstrip()
        utterance = {
            'author': author,
            'text': text
        }
        dialogue.append(utterance)
    
    topic = df.topic.values[0]
    level = df.dialog_lvl.values[0]
    link = df.youtube_link.values[0]
    trimmed_link = link.split('?t=')[0]
    
    json_list = [{
        'data': {
            'dialogue': dialogue,
            'topic': topic,
            'level': level,
            'youtube_link': trimmed_link
        }
    }]

    return json_list


def parse_txt(df):
    
    link = df.youtube_link.values[0]
    trimmed_link = link.split('?t=')[0]

    heading = f'Explaining {df.topic.values[0]} in {df.dialog_lvl.values[0]}-level conversation\nLink: {trimmed_link}\n\n'

    dialogue_str = str()
    for index, row in df.iterrows():
        dialogue_str += f'{row.role}:'
        for sentence in row.turn:
            dialogue_str += f' {sentence}'
        dialogue_str += '\n'

    return heading + dialogue_str


def parse_transcript(text):
    lines = text.split('\n')
    dialogue = []
    for line in lines:
        parts = line.split(': ', 1)
        if len(parts) > 1:    
            author = 'Explainer' if parts[0].lstrip().rstrip() == 'Explainer' else 'Explainee'
            text = parts[1].lstrip().rstrip()
        else:
            continue
        utterance = {
            'author': author,
            'text': text
        }
        dialogue.append(utterance)
    
    json_list = [{
        'data': {
            'dialogue': dialogue,
            'topic': '',
            'level': 0,
            'youtube_link': ''
        }
    }]

    return json_list


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('-json', dest='output_json',
                        action='store_true')
    
    parser.add_argument('-txt', dest='output_txt',
                        action='store_true')
    
    parser.add_argument('-extension', dest='extension',
                        action='store_true')

    args = parser.parse_args()


    path = 'WIRED/extension' if args.extension else 'WIRED/data/corpus_dialogs'
    for root, dirs, files in os.walk(path):
        for file in files:
            # print(file)
            file_name = file.split('.')[0]

            if args.extension:
                with open(os.path.join(root, file), 'r') as f:
                    text = f.read()
                json_format = parse_transcript(text)
                os.makedirs('WIRED/data/json', exist_ok=True)
                with open(f'WIRED/data/json/{file_name}_processed.json', 'w') as f:
                    f.write(json.dumps(json_format, indent=4))
                
            else:    
                df = pd.read_json(os.path.join(root, file))

                if args.output_json:
                    json_format = parse_json(df)
                    os.makedirs('WIRED/data/json', exist_ok=True)
                    with open(f'WIRED/data/json/{file_name}_processed.json', 'w') as f:
                        f.write(json.dumps(json_format, indent=4))

                if args.output_txt:
                    txt_format = parse_txt(df)
                    os.makedirs('WIRED/data/txt', exist_ok=True)
                    with open(f'WIRED/data/txt/{file_name}_processed.txt', 'w') as f:
                        f.write(txt_format)