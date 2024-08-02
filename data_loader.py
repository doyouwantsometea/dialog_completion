import pandas as pd


def load_data_file(path: str):
    df = pd.read_json(path).reset_index(drop=True)
    return df


class DataLoader(object):

    def __init__(self,
                 path: str,
                 role: str = 'Explainer',
                 turn_len: int = 100,
                 window: int = 2,
                 replace: bool = True):
        """
        Initialize DataLoader instance using a prompting configuration file.
        :param path: Path to the data file.
        :param role: Speaker of the desired turns (Explainer / Explainee).
        :param turn_len: Minimum token numbers of the desired turns.
        :param window: Number of prior and following turns to be included in the dialogue segment.
        :param replace: Whether to replace the filtered turn with {missing part} placeholder.
        """
        super().__init__()
        # load prompting configuration:
        self.df = load_data_file(path)
        self.role = role
        self.turn_len = turn_len
        self.window = window
        self.replace = replace
    
    def get_topic(self) -> str:
        """
        Parse dialogue topic.
        :return: Dialogue topic.
        """
        topic = self.df.loc[0, 'topic'].replace('_', ' ')
        return f'about {topic} '

    def get_dialog_lvl(self) -> tuple:
        """
        Parse dialogue expertise level.
        :return: Tuple that indicates expertise level of explainer and explainee.
        """
        mapping = {
            'child': (' teacher', ' child'),
            'teenager': (' teacher', ' teenager'),
            'undergrad': (' professor', ' college student'),
            'grad': (' professor', ' graduate student'),
            'colleague': ('n expert', 'nother expert')
        }
        
        level = self.df.loc[0, 'dialog_lvl']
        return mapping[level]
    
    def filter_turn(self) -> list:
        """
        Filter turns with the given conditions.
        :return: List of indexes that point toward turns in the dataframe.
        """
        df_filter = self.df[(self.df.role == self.role) & (self.df.turn_num_tokens > self.turn_len)]
        return df_filter.index
    
    def parse_diaolgue(self,
                       index: int,
                       following_context: bool) -> str:
        """
        Parse turns into dialogue segment for prompting.
        :param index: Index of the filtered turn.
        :param following_context: Whether to incorporate turns occuring after the target turn.
        :return: Dialogue string that can be applied to the prompt.
        """
        start = index - self.window if self.window <= index else 0
        if following_context:
            end = index + 1
        else:
            end = index + 1 + self.window if index + 1 + self.window <= len(self.df) else len(self.df)
        # print(self.df[start:end])

        target_turn = str()
        parsed_dialogue = str()
        
        for i, r in self.df[start:end].iterrows():
            if i == index:
                target_turn = ' '.join(r.turn).rstrip()

            parsed_dialogue += f'{r.role}:'
            if i == index and self.replace:
                parsed_dialogue += ' {missing part}\n'
            else:
                parsed_dialogue += ' ' + ' '.join(r.turn) + '\n'

        return target_turn, parsed_dialogue
    


if __name__ == "__main__":
    pass