import itertools
import nltk
import re
import pandas as pd
from pprint import pprint
import readability
import spacy
from statistics import fmean
import textstat
from tqdm import tqdm

NLP = spacy.load('en_core_web_md')

class IXQuisite():
    
    def __init__(self,
                 datapoint: dict,
                 original_dialog: bool,
                 r: int=4):
        
        self.datapoint = datapoint
        # self.text = self.txt2list()  # historical reasons mainly but also for adaptation
        # self.teaching_turn = self.bio2string()
        # self.i = self.datapoint.index
        # self.topic = self.datapoint['topic']
        # self.lvl = self.datapoint['student_role']
        self.utterance = self.datapoint['model_output']
        self.reference = self.datapoint['target_utterance']
        self.dialogue = self.datapoint['dialogue']

        # print(self.utterance)
        self.raw_text = self.raw_updated(original_dialog=original_dialog)
        
        self.token, self.pos = self.preprocessing(self.raw_text)
        self.nouns = self.noun_extraction()
        
        self.readability_res = readability.getmeasures(self.raw_text, 
                                                       lang='en')
        self.words = self.readability_res['sentence info']['words']
        
        self.r = r
    
    
    # def txt2list(self):
    #     # in each dialogue, the first line is a general
    #     # description of the dialogue, and the second
    #     # line is empty
    #     return self.datapoint['text'].split('\n')[2:]
    
    
    # def bio2string(self):
    #     # collect teaching acts in turn
    #     bio = self.datapoint['final_bio']
    #     # first sieve
    #     seq = str()
    #     previous = str()
    #     for item in bio:
    #         current = item[1][:3] if item[1] != 0 else item[0]
    #         if current != previous:
    #             seq += current
    #             seq += ' '
    #         previous = current
    #     # remove unnecessary tags
    #     seq = seq.replace('O ', '')
    #     seq = seq.replace('T09 ', '').replace('T10 ', '')
    #     # second sieve
    #     seq2 = str()
    #     previous = str()
    #     for item in seq.split(' '):
    #         if item != previous:
    #             seq2 += item
    #             seq2 += ' '
    #         previous = item
    #     return seq2
    
    
    def raw_updated(self, original_dialog=False):
        # raw_text = str()
        # toberemoved = {'Explainer: ', 'Explainee: '}
        # for sentence in self.text:
        #     for item in toberemoved:
        #         sentence = sentence.replace(item, '')
        #         raw_text += sentence
        # raw_text = self.dialogue
        # print(self.dialogue)
        if original_dialog:
            raw_text = self.dialogue.replace('{missing part}', self.reference)
        else:
            raw_text = self.dialogue.replace('{missing part}', self.utterance)
        
        raw_text = re.sub(r'\n?Explainer:', '<|endoftext|>', raw_text)
        
        return raw_text
    
    
    def preprocessing(self, txt):  # can be different things
        # general function for tokens & POS
        token, pos = list(), list()
        for t in NLP(txt):
            token.append(t.text)
            pos.append(t.pos_)
        return token, pos

    
    def noun_extraction(self):
        # match text to POS
        tok2pos = dict(zip(self.token, self.pos))
        # collect nouns
        nouns = list()
        for word, pos in tok2pos.items():
            if pos == 'NOUN':
                nouns.append(word)
        return nouns
    
    
    def apply_rounding(self, v):
        return v if not self.r else round(v, self.r)
    
        
    def minimal_explanations(self):
        # frequency of NE -- other functions consider NPs
        named_entities = NLP(self.raw_text)
        return self.apply_rounding(len(named_entities.ents) 
                                   / len(self.token)) * 10
    
    
    def lexical_complexity(self):
        return self.apply_rounding(
            textstat.difficult_words(self.raw_text) / self.words) * 10

        
    def synonym_density(self):
        # frequency of synonyms
        topic = self.datapoint['file'].replace('_', ' ')[:-7]
        if topic == 'origani':
            topic = 'origami'
        topic = NLP(topic) 
        synonyms = list()
        for noun in self.nouns:
            similarity = topic.similarity(NLP(noun))
            if similarity >= 0.5: 
                synonyms.append(noun)
        return self.apply_rounding(len(synonyms) / len(self.nouns))
        
    
    def coherence(self):
        # readability: [word usage] conj, [sentence beginnings] conj, subord.
        conj_1 = self.readability_res['word usage']['conjunction']
        conj_2 = self.readability_res['sentence beginnings']['conjunction']
        sub = self.readability_res['sentence beginnings']['subordination']
        return self.apply_rounding((conj_1 + conj_2 + sub) / self.words)
    
    
    def reading_grade(self):
        # readability grades: Flesch-Kincaid
        return self.apply_rounding(textstat.flesch_kincaid_grade(self.raw_text) 
                                   / 18)
    
    def adaptation(self):
        # pairwise similarity between nouns
        similarities = list()
        for a, b in itertools.combinations(self.nouns, 2):
            similarities.append(NLP(a).similarity(NLP(b)))
        return self.apply_rounding(fmean(similarities))
        
 
    # def teaching_model(self):
    #     # using inverse edit distance.
    #     # teaching acts should ideally come in the right order:
    #     return self.apply_rounding(1 / nltk.edit_distance(
    #                                 "T01 T02 T03 T04 T05 T06 T07 T08",
    #                                 self.teaching_turn))
        
    
    # DEPRECATED
    # def readability_2(self):
    #     # readability grades: Flesch Reading Ease
    #     return self.apply_rounding(textstat.flesch_reading_ease(self.raw_text) 
    #                                / 100)

    # DEPRECATED
    # def adaptation(self):
    #     # TODO run again
    #     # count of question marks on the explainee side (!)
    #     question_marks = 0
    #     for utterance in self.text:
    #         if utterance[0] != 'E': # removes explainer!
    #             question_marks += utterance.count('?')
    #     return self.apply_rounding(question_marks / len(self.token))
    
    
 

if __name__ == "__main__":
    
    # Prepare output file
    output_json = {
        # 'topic': dict(),
        # 'lvl': dict(),
        'minimal_explanations': dict(),
        'lexical_complexity': dict(),
        'synonym_density': dict(),
        'coherence': dict(),
        'reading_grade': dict(),
        'adaptation': dict(),
        # 'teaching_model': dict()
        }
    
    # Open and preprocess input files
    # file_path = '../../data/final_annotation/ta_full.jsonl'
    # file_path = 'WIRED/data/corpus_dialogs/blackhole_3.json'
    file_path = 'results/WIRED_Meta-Llama-3-8B-Instruct_l60_w2.json'
    ta_data = pd.read_json(path_or_buf=file_path)
    
    print(ta_data.head())
    # Index(['id', 'topic', 'student_role', 'text', 'final_bio'], dtype='object')
    # for index, row in ta_data.iterrows():
    for index, row in tqdm(ta_data.iterrows(),  total=ta_data.shape[0], desc=f'Processing DF'):

        ts = IXQuisite(datapoint=row.to_dict(),
                       original_dialog=False,
                       r=4)
        # i = ts.i
        # print(i)
        # output_json['topic'][index] = ts.topic
        # output_json['lvl'][index] = ts.lvl
        output_json['minimal_explanations'][index] = ts.minimal_explanations()
        output_json['lexical_complexity'][index] = ts.lexical_complexity()
        output_json['synonym_density'][index] = ts.synonym_density()
        output_json['coherence'][index] = ts.coherence()
        output_json['reading_grade'][index] = ts.reading_grade()
        output_json['adaptation'][index] = ts.adaptation()
        # output_json['teaching_model'][i] = ts.teaching_model()
        print(output_json)

    # out = pd.DataFrame.from_dict(output_json).to_csv('testsuite_v4.csv')
    print(output_json)
