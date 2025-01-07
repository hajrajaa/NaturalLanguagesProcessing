#%%
import os
import math
import random
import sys
import re
import json
import numpy as np
import pandas as pd 
from gensim.models import Word2Vec



# section 1:1 - Tokenization 
def tokenize_sentences(sentences):

    tokenized_sentences=[]
    
    try:

        for sentence in sentences:

           
            sentence=re.sub(r"[-_]"," ",sentence)
            curr_toknes=re.sub(r"[!().:;,?%!]","",sentence).split()
            
            filtered_toknes=[]
            for token in curr_toknes:

                token=token.strip()

                if len(token)<2 :
                    continue
                
                elif any(char.isdigit() for char in token):
                    continue

                filtered_toknes.append(token)


            print(filtered_toknes)
                
                
                
            tokenized_sentences.append(filtered_toknes)


    except Exception as e:
        raise e
    
    return tokenized_sentences



# Load the corpus
def load_corpus(file_path):

    corpus=[]
    try:
        with open(file_path ,'r', encoding='utf-8') as f:

            for sentence in f:
                # get the data from the json file 
                data=json.loads(sentence)
                corpus.append(data)

            if not corpus:
                raise ValueError("Empty corpus")
            
    except Exception as e :
        raise e
    
    return corpus


#%% main
if __name__=='__main__':
    print("aaaaaaaaaaa")
    try:
        
        input_path='knesset_corpus.jsonl'
        
        corpus=load_corpus(input_path) 
        
        corpus_sentences=[sentence.get('sentence_text','') for sentence in corpus]
        corpus_sentences=corpus_sentences[:10]

        toknes=tokenize_sentences(corpus_sentences)
        #print(toknes)
        
        
   
    except Exception as e:
        raise e
    

#%%