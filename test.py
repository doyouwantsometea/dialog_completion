import pandas as pd
import os
import fed
from data_loader import DataLoader
from prompter import Prompter




for root, dirs, files in os.walk('WIRED/data/corpus_dialogs'):
    for file in files:
        # print(file)

        df = pd.read_json(os.path.join(root, file))
        # print(df.head())


data_loader = DataLoader(path='WIRED/data/corpus_dialogs/blackhole_5.json',
                         role='Explainer',
                         utterance_len=100,
                         window=2,
                         replace=True)

prompter = Prompter(prompt_cfg_filename='prompts.json')



index_list = data_loader.filter_utternace()
for index in index_list:
    diaolgue = data_loader.parse_diaolgue(index=index)
    # print(diaolgue)
    prompter.build_prompt(diaolgue)



model, tokenizer = fed.load_models('microsoft/DialoGPT-large')

# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
conversation = "<|endoftext|> So far, what do you know about black holes? <|endoftext|> I never knew beforehand how hard it was to get actual data of the black holes itself, first of all, they're dark, and, like, they're so far away, it's almost impossible just to get a good image of them. They were discussing a project in which multiple radio telescopes of some sort, like, are, like, pinpointed all across, from Greenland to South America, and, like, and they're trying to get an image of the black hole in the center of our galaxy because, as opposed to just recording its impact on the surrounding stars and planets. <|endoftext|> So we've been, we've had, now, effectively two different ways of getting more direct measurements, one is the LIGO, which is the Laser Interferometer Gravitational Wave Observatory, which is where, getting the ripples in space time, coming off of the merging of black holes. The other one that you're mentioning is actually called the Event Horizon Telescope, where they're using radio waves to actually image the event horizon, that region where light cannot escape from the black hole at the center of our galaxy, which I know they're working on it right now. It's an amazing thing, but that'll be the most direct imaging of a black hole. LIGO is a direct detection of the consequence of the merging of black holes. The critical part has been, like, for the super massive black hole at the center of our galaxy, we've seen the stars orbiting it, and we've measured the mass, so that way, so if you look at a spinning black hole, it actually fundamentally alters the emission that's coming off the stuff that's falling into it. These are discovered as what are called X-ray binaries, that is, you know, there's an X-ray member of the binary that is emitting in the X-rays, and it's really not very bright in the optical (mumbles) at all, so there's always, people are looking at these X-ray binaries. <|endoftext|> What sort of technology and, like, I guess tools have you been using in your studies, or, like, just in general, in the study of black holes? <|endoftext|> For my studies, I actually, when I started at UCLA in graduate school, I worked with a professor named Matt Malkin who was, gotten a lot of data observations from the Hubble Space Telescope, so that was one of my very first projects to work on, so any, space-based observatories have been a really big advantage, and then I've moved on now to the Spitzer Space Telescope. In addition to that, then there's other people who've used a lot of X-ray telescopes, NuSTAR, Chandra have used data from that. It's been a combination of both ground-based observatories, as well as space-based ones, and going everywhere from X-ray observations, not done by me, but certainly ultraviolet, and then optical, and infrared, particularly, those are the ones that I've been most involved with."
scores = fed.evaluate(conversation,
                      model,
                      tokenizer)

print(scores)

# text = str()

# for utterance in df[2]['dialog']:
#     text += (utterance['Sentence'][0] + ' ')

# f = open('test.txt', 'a')
# f.write(text)
# f.close()