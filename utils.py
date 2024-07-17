import json
import re



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
