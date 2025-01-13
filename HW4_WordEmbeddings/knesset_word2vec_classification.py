#%%
import os
import math
import random
import sys
import json
import numpy as np
import pandas as pd 
from collections import defaultdict,Counter

from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score
from gensim.models import Word2Vec


random.seed(42)
np.random.seed(42)

class speakerData:

    def __init__(self,name):
        self.name=name
        self.sentences=[]    #list to store all the sentences of the speaker
        self.sentences_count=0   # number of sentences of the speaker       
    
    def add_sentence(self,sentence):
        self.sentences.append(sentence)
        self.sentences_count+=1   # update the number of sentences 
###########################################################################################################################################

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

def load_sentences(file_path):
    sentences=[]
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                
                sentences.append(line.strip())
    except Exception as e:
        raise e
    
    return sentences


def get_common_speakers(corpus):
    
    # counter for the speakers 
    speaker_counter=Counter()

    for data in corpus:

        speaker_name=data.get("speaker_name",None)
        if speaker_name:
            speaker_counter[speaker_name]+=1

    # get the most 2 common speakers
    most_common_speakers=speaker_counter.most_common(2)

    #get the names of the speakers 
    first_speaker_name=most_common_speakers[0][0] if len(most_common_speakers)>0 else None
    secound_speaker_name=most_common_speakers[1][0] if len(most_common_speakers)>1 else None

    return first_speaker_name,secound_speaker_name 


# split the data to the speakers classes 
def split_data(first_speaker,secound_speaker,corpus):

    # aliased for the speakers name 
    aliases={
        first_speaker.name:{"ר' ריבלין", "רובי ריבלין"},
             secound_speaker.name: {"אברהם בורג"}
    }

    for data in corpus:
        
        curr_name=data.get("speaker_name",None)

        # in case of the curr name is the first speaker name or is an alias of the first_speaker name 
        if curr_name in aliases.get(first_speaker.name,set()) or curr_name==first_speaker.name:
            ## add the sentence to the first speaker class 
            first_speaker.add_sentence(data)

        ## in case of the curr name is the second speaker name or is an alias of the first_speaker name 
        elif curr_name in aliases.get(secound_speaker.name,set()) or curr_name==secound_speaker.name :
            # add the sentence to the second speaker class 
            secound_speaker.add_sentence(data)


    return first_speaker,secound_speaker


def down_sample(first_speaker,secound_speaker):

    # get the size of the speakers classes 
    first_speaker_size=first_speaker.sentences_count
    secound_speaker_size=secound_speaker.sentences_count

    # in case the size of the first speaker grather than the secound speaker 
    if first_speaker_size>secound_speaker_size:
        first_speaker.sentences=random.sample(first_speaker.sentences,secound_speaker_size)
        # update the size field
        first_speaker.sentences_count=secound_speaker_size

    # in case the size of the second speaker grather than the first speaker     
    elif first_speaker_size<secound_speaker_size:
        secound_speaker.sentences=random.sample(secound_speaker.sentences,first_speaker_size)
        # update the size field
        secound_speaker.sentences_count=first_speaker_size   

    return first_speaker,secound_speaker





# train the models 
def train_model(features,labels):
    
    cross_val=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

    #KNN model
    knn_model=KNeighborsClassifier(n_neighbors=5)
    knn_cv_scores=cross_val_score(knn_model,features,labels,cv=cross_val,scoring='accuracy')
    knn_model.fit(features,labels)

    return knn_model,knn_cv_scores

def run_training(sentences,labels,word_vectors):
    try:

        
        sentences_txt=[sentence.get('sentence_text','') for sentence in sentences]

        sentences_embeddings=get_sentence_embeddings(sentences_txt,word_vectors)
        sentences_embeddings=np.array(sentences_embeddings)
        labels=np.array(labels)
      
        # dictionary to store the classification report 
        report={}

        # train the models 
        knn_model,knn_cv_scores=train_model(sentences_embeddings,labels)
        report['KNN']=classification_report(labels,knn_model.predict(sentences_embeddings), output_dict=True)
        
        
        
        print("\nKNN Classification Report:\n")
        print(pd.DataFrame(report).transpose())
        print("\n")


        return knn_model

    except Exception as e:
        raise e
    

#############################################################################################################################

def get_sentence_embeddings(sentences,word_vectors):

    sentences_embeddings=[]

    for sentence in sentences:
        #sentence=sentence.split()
        sentence_vector=np.zeros(word_vectors.vector_size)
        k=0   # number of words in the sentence 
        for token in sentence:
            # to ensure that the token is in the word vectors
            if token in word_vectors.key_to_index.keys():
                sentence_vector+=word_vectors[token]
                k+=1

            if k>0 :
                sentence_vector/=k
        sentences_embeddings.append(sentence_vector)  
    return sentences_embeddings       

def run_main(input_path,model_path):

     # load the corpus
    corpus=load_corpus(input_path) 

    # get the most common speakers 
    first_speaker_name,secound_speaker_name=get_common_speakers(corpus)

    first_speaker=speakerData(first_speaker_name)
    secound_speaker=speakerData(secound_speaker_name)
   

    # section 1 - split the data 
    first_speaker,secound_speaker=split_data(first_speaker,secound_speaker,corpus)

    # section 3- down sample the data 
    first_speaker,secound_speaker=down_sample(first_speaker,secound_speaker)

    binary_sentences=first_speaker.sentences+secound_speaker.sentences

    

    # labels for the binary and multu-class classification 
    labels_binary=["first"]*first_speaker.sentences_count+["second"]*secound_speaker.sentences_count
    labels_binary=np.array(labels_binary)
    

    
    word_vectors=Word2Vec.load(model_path).wv
    

    # section 4 - train the model
    knn_model=run_training(binary_sentences,labels_binary,word_vectors)
    
    # # section 5 - classify the speaker 
    # test_sentences=load_sentences(sentences_texts_file)
    # classify_speaker(test_sentences,output_dir,knn_model)


    
     
   


# %% Main 

if __name__=='__main__':

    try:
        # if len(sys.argv)!=4:
        #     raise ValueError("Invalid Input Arguments!!")
        
        # input_path=sys.argv[1]

        # if not os.path.exists(input_path):
        #     raise FileNotFoundError(f"File Not Found:{input_path}")
        
        # sentences_texts_file=sys.argv[2]

        # if not os.path.exists(sentences_texts_file):
        #     raise FileNotFoundError(f"File Not Found:{sentences_texts_file}")

        # output_dir=sys.argv[3]

        # if not os.path.exists(output_dir):
        #     raise FileNotFoundError(f"File Not Found:{output_dir}")

        input_path='knesset_corpus.jsonl'
        model_path='knesset_word2vec.model'

        
        
        run_main(input_path,model_path) 
        
        
   
    except Exception as e:
        raise e
    





# %%