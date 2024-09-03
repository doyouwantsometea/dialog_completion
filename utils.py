import json
import re
import heapq
import pandas as pd



def extract_json(text: str):
    # Regular expression to match a JSON-formatted string
    json_pattern = r'\{.*?\}'
    
    # Find the JSON string in the text
    match = re.search(json_pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            # Parse the JSON string to ensure it is valid
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError:
            return None
    return None


def trim_after_placeholder(text: str,
                           placeholder: str):

    index = text.find(placeholder)

    # Trim everything after the key phrase
    if index != -1:
        return text[:index + len(placeholder)].replace(placeholder, '{"missing_part": }')
    else:
        return None


def flatten_dialogue(dialogue: str,
                     reference: str,
                     model_turn: str,
                     original_dialog: bool=False):
        if original_dialog:
            raw_text = dialogue.replace('{missing part}', reference)
        else:
            raw_text = dialogue.replace('{missing part}', model_turn)
        
        raw_text = re.sub(r'\n?(Explainer|Explainee):', '<|endoftext|>', raw_text)
        
        return raw_text


def remove_training_set(files):
    return [file for file in files if 'train' not in file]


def get_worst_features(row, n):
    pairs = zip(row.index, row.values)
    return heapq.nsmallest(n, pairs, key=lambda x: x[1])