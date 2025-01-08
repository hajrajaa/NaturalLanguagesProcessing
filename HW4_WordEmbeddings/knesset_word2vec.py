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


            #print(filtered_toknes)
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


def get_similar_words(words_list,word_vectors,output_dir='output'):

    similar_words={}
    #print(words_list)
    try:
        #output_path=os.path.join(output_dir,"knesset_similar_words.txt")
        with open("knesset_similar_words.txt",'w',encoding='utf-8') as file:
            for word in words_list:
                print(word)
                similarity_scores={other_word: word_vectors.similarity(word,other_word) for other_word in word_vectors.key_to_index if other_word!=word }
                
                sorted_similarty_scores=sorted(similarity_scores.items(),key=lambda item: item[1], reverse=True)
                file.write(f"{word}: ")
                for w,score in sorted_similarty_scores[:4]:
                    file.write(f"({w},{score}),")
                last_word,last_score=sorted_similarty_scores[4]
                file.write(f"({last_word},{last_score})\n")

                ## check if we need to return the similarty score . 
                #similar_words[word]=sorted_similarty_scores


    except Exception as e :
        raise e
    
    return similar_words


#%% main
if __name__=='__main__':
    print("aaaaaaaaaaa")
    try:
        
        input_path='knesset_corpus.jsonl'
        
        corpus=load_corpus(input_path) 
        
        corpus_sentences=[sentence.get('sentence_text','') for sentence in corpus]
        #corpus_sentences=corpus_sentences[:10]

        ###########################################################################################################################################
        # section 1:1
        toknes=tokenize_sentences(corpus_sentences)
        #print(toknes)

        # section 1:2 
        # build the word2vec model
        model=Word2Vec(sentences=toknes,vector_size=50,window=50,min_count=1)

        # save the trained model
        model.save('knesset_word2vec.model')

        #section 1:3 
        word_vectors=model.wv
        print(word_vectors['ישראל'])   # get the vector of the word 
        ##################################################################################################################################################
        # section2 - similarity between words 
        words_list=[
            'ישראל','גברת','ממשלה','חבר','בוקר','מים','אסור','רשות','זכויות'
        ]
        # to do : fix the path for the output 
        similar_words=get_similar_words(words_list,word_vectors)
       
             


   
    except Exception as e:
        raise e
    

#%%