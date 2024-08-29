import json
import pandas as pd
from utils import trim_after_placeholder


def load_prompt_config(config_name: str,
                       task: str) -> dict:
    """
    Load a prompting configuration file.
    :param config_name: Name of the prompting configuration file.
    :return: Dictionary prompting configuration of the given task.
    """
    with open(config_name, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)[task]
    return config


class Prompter(object):
    """
    Class to build full prompts based on a prompting configuration.
    """
    def __init__(self,
                 prompt_cfg_filename: str,
                 task: str):
        """
        Initialize Prompter instance using a prompting configuration file.
        :param prompt_cfg_filename: Name of the prompting configuration file.
        :param task: Task name to be prompting configuration file.
        """
        super().__init__()
        # load prompting configuration:
        self.cfg = load_prompt_config(prompt_cfg_filename, task)

    def __call__(self, dialogue):
        """
        Convenience method passing all arguments to build_prompt() method.
        :param footer_idx: Sample index for prompt footer data.
        :return: Fully built prompt string based on prompting configuration.
        """
        return self.build_prompt(dialogue=dialogue)

    # not sure if this is needed here
    def build_one_shot_prompt(self):
        prompt = str()
        if 'one_shot_instruction' in self.cfg:
            prompt += self.cfg['one_shot_instruction']
        return prompt

    def build_prompt(self,
                     dialogue: str,
                     topic: str = '',
                     explainer: str = 'n explainer',
                     explainee: str = 'n explainee',
                     footer_context: bool = False,
                     instruction: str = '') -> str:
        """
        Build full prompt based on prompting configuration and footer sample index.
        :param topic: Mentioning topic of explanatory dialogue in the prompt (optional).
        :param explainer: Description of the explaner participating in the dialogue.
        :param explainee: Description of the explanee participating in the dialogue.
        :param footer_context: Whether provide dialogue context in the footer.
        :param instruction: Perturbation for tuning model-generated dialogue.
        :return: Fully built prompt string based on prompting configuration.
        """
        
        prompt = str()
        
        # add header
        prompt += self.cfg['header']
        prompt = prompt.replace('{topic}', topic).replace('{explainer}', explainer).replace('{explainee}', explainee)
        
        # dialogue
        prompt += f'{dialogue}\n'

        #TODO: prompt += instruction

        # add footer:
        if footer_context:
            prompt += self.cfg['footer_w_context']
            prompt += trim_after_placeholder(text=dialogue,
                                             placeholder='{missing part}')
        else:
            prompt += self.cfg['footer']
        
        return prompt


if __name__ == "__main__":
    pass