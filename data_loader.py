import pandas as pd


def load_data_file(path: str):
    df = pd.read_json(path).reset_index(drop=True)
    return df


class DataLoader(object):

    def __init__(self,
                 path: str,
                 role: str = 'Explainer',
                 utterance_len: int = 100,
                 window: int = 2,
                 replace: bool = True):
        """
        Initialize DataLoader instance using a prompting configuration file.
        :param path: Path to the data file.
        :param role: Speaker of the desired utterances (Explainer / Explainee).
        :param utterance_len: Minimum token numbers of the desired utterances.
        :param window: Number of prior and following utterances to be included in the dialogue segment.
        :param replace: Whether to replace the filtered utterance with {missing part} placeholder.
        """
        super().__init__()
        # load prompting configuration:
        self.df = load_data_file(path)
        self.role = role
        self.utterance_len = utterance_len
        self.window = window
        self.replace = replace
    
    def filter_utternace(self) -> list:
        """
        Filter utterances with the given conditions.
        :return: List of indexes that point toward utterances in the dataframe.
        """
        df_filter = self.df[(self.df.role == self.role) & (self.df.turn_num_tokens > self.utterance_len)]
        return df_filter.index
    
    def parse_diaolgue(self,
                       index: int) -> str:
        """
        Parse utterances into dialogue segment for prompting.
        :param index: Index of the filtered utterance.
        :return: Dialogue string that can be applied to the prompt.
        """
        start = index - self.window if self.window <= index else 0
        end = index + 1 + self.window if index + 1 + self.window <= len(self.df) else len(self.df)
        # print(self.df[start:end])

        target_utterance = str()
        parsed_dialogue = str()
        
        for i, r in self.df[start:end].iterrows():
            if i == index:
                target_utterance = ' '.join(r.turn).rstrip()

            parsed_dialogue += f'{r.role}:'
            if i == index and self.replace:
                parsed_dialogue += ' {missing part}\n'
            else:
                parsed_dialogue += ' ' + ' '.join(r.turn) + '\n'

        return target_utterance, parsed_dialogue
    





if __name__ == "__main__":
    pass