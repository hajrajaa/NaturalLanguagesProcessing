#%%
import os
import math
import random
import sys
import json
import numpy as np
import pandas as pd 
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


random.seed(42)
np.random.seed(42)

class speakerData:

    def __init__(self,name):
        self.name=name
        self.sentences=[]    #list to store all the sentences of the speaker
        self.sentences_count=0
        self.bow_vectors=None #to store the BoW vectors of the speaker's sentences
    
    def add_sentence(self,sentence):
        self.sentences.append(sentence)
        self.sentences_count+=1

    ## SECTION 3 :1 - Extract BoW feature vector
    def extract_BoW_vector(self , vectorizer=None):

        if not self.sentences:
            raise ValueError(f"No sentences for  speaker {self.name}")
        
        if not vectorizer:
            vectorizer=CountVectorizer()

        texts=[sentence['sentence_text'] for sentence in self.sentences if 'sentence_text' in sentence]
        self.bow_vectors=vectorizer.fit_transform(texts)

        return vectorizer




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



## section 2 
def down_sample(first_speaker,secound_speaker):

    first_speaker_size=first_speaker.sentences_count
    secound_speaker_size=secound_speaker.sentences_count

    print(f"First Speaker Size before: {first_speaker_size}")
    print(f"Secound Speaker Size before: {secound_speaker_size}")

    if first_speaker_size>secound_speaker_size:
        first_speaker.sentences=random.sample(first_speaker.sentences,secound_speaker_size)
        first_speaker.sentences_count=secound_speaker_size
    elif first_speaker_size<secound_speaker_size:
        secound_speaker.sentences=random.sample(secound_speaker.sentences,first_speaker_size)
        secound_speaker.sentences_count=first_speaker_size

    print(f"First Speaker Size after: {first_speaker.sentences_count}")
    print(f"Secound Speaker Size after: {secound_speaker.sentences_count}")

    return first_speaker,secound_speaker



def extract_feature_vector(sentences):

    features=[]

    # label_encoder=LabelEncoder()
    # encoded_speakers=label_encoder.fit_transform([sentence['speaker_name'] for sentence in sentences])

    for i,sentence in enumerate(sentences):

        feature_vector={}

        # first feature - protocol type 
        #feature_vector['speaker_name']=encoded_speakers[i]
        feature_vector['protocol_type']=sentence.get('protocol_type',None)

        # secound feature - average word length per sentence
        words=sentence.get('sentence_text',None).split()
        avg_word_length=sum(len(w) for w in words)/len(words) if words else 0 
        feature_vector['avg_word_length']=avg_word_length

        # third feature - number of words in the sentence
        feature_vector['words_num']=len(words)


        #forth feature- number of unique words in the sentence
        unique_words=set(words)
        feature_vector['unique_words_num']=len(unique_words)


        #fifth feature - number of punctuations marks in the sentence
        punctuations="'!%().,:;?'-"
        punctuations_count=sum(1 for w in words if w in punctuations)
        feature_vector['punctuations_count']=punctuations_count

        # sixst feature 

        committee_collocations=[('זה','את'),('רוצה','אני'),('דבר','של','בסופו')]
        plenary_collocations=[('הכנסת','חבר'),('נוכח','אינו'),('ראש',',')]

        all_collocations=committee_collocations+plenary_collocations

        for collocation in all_collocations:
            collocation_str=''.join(collocation)
            feature_vector[f'collocation_{collocation_str}']=sentence.get('sentence_text','').count(collocation_str)

        features.append(feature_vector)

    return features

    


# %% Main 

if __name__=='__main__':

    try:
        input_file='knesset_corpus.jsonl'
        corpus=load_corpus(input_file)

        
        # section 1 
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

        # section 2 - down sample the data 
        first_speaker,secound_speaker=down_sample(first_speaker,secound_speaker)
        
        first_speaker,other_spekaers=down_sample(first_speaker,other_spekaers)

        if other_spekaers.sentences_count!=secound_speaker.sentences_count:
            secound_speaker=down_sample(secound_speaker,other_spekaers)

        print("First Speaker")
        for sentence in first_speaker.sentences:
            print(sentence.get("sentence_text",None))
            print("\n")
       
        for sentence in secound_speaker.sentences:
            print(sentence.get("sentence_text",None))
            print("\n")
        



        # section 3:1 _ extract the BoW feature vectors
        vectorizer=CountVectorizer()
        first_speaker.extract_BoW_vector(vectorizer)
        secound_speaker.extract_BoW_vector(vectorizer)

        #print((first_speaker.bow_vectors.toarray()[1]))
        

              
   
    except Exception as e:
        raise e




# %%