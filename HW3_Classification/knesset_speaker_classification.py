#%%
import os
import math
import random
import pandas as pd 
from collections import defaultdict,Counter
import sys

import json

class speakerData:

    def __init__(self,name):
        self.name=name
        self.sentences=[]    # list to store all the sentences of the speaker
        self.sentences_count=0
    
    def add_sentence(self,sentence):
        self.sentences.append(sentence)
        self.sentences_count+=1





def load_corpus(file_path):

    corpus=[]
    try:
        with open(file_path ,'r', encoding='utf-8') as f:

            for sentence in f:
                data=json.loads(sentence)
                corpus.append(data)

            if not corpus:
                raise ValueError("Empty corpus")
            
    except Exception as e :
        raise e
    
    return corpus




# %% Main 

if __name__=='__main__':

    try:
        input_file='knesset_corpus.jsonl'
        corpus=load_corpus(input_file)

        

        speaker_counter=Counter()
        for data in corpus:

            speaker_name=data.get("speaker_name",None)
            if speaker_name:
                speaker_counter[speaker_name]+=1

        # get the most common speakers
        most_common_speakers=speaker_counter.most_common(2)
        


        first_speaker_name=most_common_speakers[0][0] if len(most_common_speakers)>0 else None
        secound_speaker_name=most_common_speakers[1][0] if len(most_common_speakers)>1 else None

        print(f"First Speaker: {first_speaker_name}")
        print(f"Secound Speaker: {secound_speaker_name}")

        first_speaker=speakerData(first_speaker_name)
        secound_speaker=speakerData(secound_speaker_name)
        other_spekaers=speakerData("other")

        for data in corpus:
          
           curr_name=data.get("speaker_name",None)
           
           if curr_name==first_speaker_name:
               first_speaker.add_sentence(data)
           elif curr_name==secound_speaker_name:
               secound_speaker.add_sentence(data)
           else:
                other_spekaers.add_sentence(data)
              
   
    except Exception as e:
        raise e




# %%