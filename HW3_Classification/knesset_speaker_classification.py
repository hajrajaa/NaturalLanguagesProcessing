#%%
import os
import math
import random
import sys
import json
import numpy as np
import pandas as pd 
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score


random.seed(42)
np.random.seed(42)

class speakerData:

    def __init__(self,name):
        self.name=name
        self.sentences=[]    #list to store all the sentences of the speaker
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



## section 2 
def down_sample(first_speaker,secound_speaker):

    first_speaker_size=first_speaker.sentences_count
    secound_speaker_size=secound_speaker.sentences_count

    # print(f"First Speaker Size before: {first_speaker_size}")
    # print(f"Secound Speaker Size before: {secound_speaker_size}")

    if first_speaker_size>secound_speaker_size:
        first_speaker.sentences=random.sample(first_speaker.sentences,secound_speaker_size)
        first_speaker.sentences_count=secound_speaker_size
    elif first_speaker_size<secound_speaker_size:
        secound_speaker.sentences=random.sample(secound_speaker.sentences,first_speaker_size)
        secound_speaker.sentences_count=first_speaker_size

    # print(f"First Speaker Size after: {first_speaker.sentences_count}")
    # print(f"Secound Speaker Size after: {secound_speaker.sentences_count}")

    return first_speaker,secound_speaker



def extract_feature_vector(sentences):

    features=[]
  

    for sentence in sentences:

        feature_vector={}

        #fisrt feature - sentence length
        curr_sentence_txt=sentence.get('sentence_text','')
       
        feature_vector['sentence_length']=len(curr_sentence_txt)

        #secound faeture- protocol number
        protocol_number=sentence.get('protocol_number',None)
        feature_vector['protocol_number']=protocol_number

        #third feature- knesset number
        knesset_number=sentence.get('knesset_number',None)
        feature_vector['knesset_number']=knesset_number


    
        # fourth feature - average word length per sentence
        words=curr_sentence_txt.split()
        avg_word_length=sum(len(w) for w in words)/len(words) if words else 0 
        feature_vector['avg_word_length']=avg_word_length

        # fifth feature - number of words in the sentence
        feature_vector['words_num']=len(words)


        #sixst feature- number of unique words in the sentence
        unique_words=set(words)
        feature_vector['unique_words_num']=len(unique_words)

     
        #seventh feature - number of punctuations marks in the sentence
        punctuations=r"'!().,;?-\":-%"
        punctuations_count=sum(1 for char in curr_sentence_txt if char in punctuations )
        feature_vector['punctuations_count']=punctuations_count

        
        #eight feature - special words 
        words=[('ההסתייגות'),('הצבעה'),('סעיף')]
        #words=[('ההסתייגות'),('סעיף'),('הוועדה'),('קבוצת'),('הצבעה'),('אינו'),('בעד'),('נגד'),('אדוני'),('אני'),('נוכח'),('ממשלה'),('חבר'),('הכנסת')]
        for word in words:
            feature_vector[f'special_word_{word}']=curr_sentence_txt.count(word)

        features.append(feature_vector)

        

    return pd.DataFrame(features)



def train_model(features,labels):
    
    cross_val=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

    #KNN model
    #knn=KNeighborsClassifier(n_neighbors=10)
    knn_model=KNeighborsClassifier(n_neighbors=5)
    knn_cv_scores=cross_val_score(knn_model,features,labels,cv=cross_val,scoring='accuracy')
    knn_model.fit(features,labels)
    # print("KNN Cross Validation Scores: ",knn_cv_scores)
    # print("KNN Average Cross Validation Score: ",np.mean(knn_cv_scores))

    #Logistic Regression model

    lr_model=LogisticRegression(max_iter=10000)
    
    lr_cv_scores=cross_val_score(lr_model,features,labels,cv=cross_val,scoring='accuracy')
    lr_model.fit(features,labels)
    # print("Logistic Regression Cross Validation Scores: ",lr_cv_scores)
    # print("Logistic Regression Average Cross Validation Score: ",np.mean(lr_cv_scores))

    return knn_model,knn_cv_scores.mean(),lr_model,lr_cv_scores.mean()


# ## SECTION 3 :1 - Extract BoW feature vector
def extract_BoW_vector(vectorizer=None,sentences=None):

    if not sentences:
        raise ValueError(f"No sentences ")
    
    bow_vectors=vectorizer.fit_transform(sentences)

    return bow_vectors

# Section 5 - classify the speakers 
def classify_speaker(sentences,output_file,model,vectorizer=None,feature_type='bow'):

    try:
        with open(output_file,'w',encoding='utf-8')as f:

            sentences_features=vectorizer.transform(sentences)
        
            # if feature_type=='bow':
            #     sentences_features=vectorizer.transform(sentences)
                
            # else:
            #     sentences_features=extract_feature_vector(sentences)

            
            predictions=model.predict(sentences_features)
           

            for pred in predictions:
                f.write(json.dumps(pred)+'\n')


    except Exception as e:
        raise e
    
        


def load_sentences(file_path):
    sentences=[]
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                sentences.append(line.strip())
    except Exception as e:
        raise e
    
    return sentences




def run_training(sentences,labels,classification_type='Multi') :
    try:
        sentences_txt=[sentence.get('sentence_text','') for sentence in sentences]
        
        vectorizer_tf=TfidfVectorizer()
        bow_vector=extract_BoW_vector(vectorizer_tf,sentences_txt)
        features_vector=extract_feature_vector(sentences)
    
        print(f"{classification_type} Classification:")
        knn_model,bow_knn,lr_model,bow_lr=train_model(bow_vector,labels)
        print("BoW Classifier Score -knn:  ",bow_knn)
        print("BoW Classifier Score-lr: ",bow_lr)
        custom_knn_model,feature_knn,custom_lr_model,feature_lr=train_model(features_vector,labels)
        print("Custom Feature Classifier Score -knn: ",feature_knn)
        print("Custom Feature Classifier Score -lr: ",feature_lr)

        return knn_model,custom_knn_model,lr_model,custom_lr_model,vectorizer_tf



    except Exception as e:
        raise e

# section 1 - get the most common speakers
def get_common_speakers(corpus):
    
    speaker_counter=Counter()
    for data in corpus:

        speaker_name=data.get("speaker_name",None)
        if speaker_name:
            speaker_counter[speaker_name]+=1

    # get the most common speakers
    most_common_speakers=speaker_counter.most_common(2)
    first_speaker_name=most_common_speakers[0][0] if len(most_common_speakers)>0 else None
    secound_speaker_name=most_common_speakers[1][0] if len(most_common_speakers)>1 else None

    # print(f"First Speaker: {first_speaker_name}")
    # print(f"Secound Speaker: {secound_speaker_name}")

    return first_speaker_name,secound_speaker_name 


def split_data(first_speaker,secound_speaker,other_speakers,corpus):

    aliases={first_speaker.name:{"ר' ריבלין", "רובי ריבלין", first_speaker.name},
             secound_speaker.name: {"אברהם בורג" ,secound_speaker.name}
             }

    for data in corpus:
        
        curr_name=data.get("speaker_name",None)
        
        if curr_name in aliases[first_speaker.name]:
            first_speaker.add_sentence(data)
        elif curr_name in aliases[secound_speaker_name]:
            secound_speaker.add_sentence(data)
        else:
            other_speakers.add_sentence(data)

    return first_speaker,secound_speaker,other_speakers







    



    


# %% Main 

if __name__=='__main__':

    try:
        input_file='knesset_corpus.jsonl'
        corpus=load_corpus(input_file)

        sentences_path='knesset_sentences.txt'

        first_speaker_name,secound_speaker_name=get_common_speakers(corpus)

        first_speaker=speakerData(first_speaker_name)
        secound_speaker=speakerData(secound_speaker_name)
        other_speakers=speakerData("other")

        # section 1 - split the data 
        first_speaker,secound_speaker,other_speakers=split_data(first_speaker,secound_speaker,other_speakers,corpus)
    
        
        # section 2 - down sample the data 
        print("DOWN SAMPLING THE DATA:\n")
        print("size before down sampling:\n")
        print(f"Size of the First Speaker:{first_speaker.sentences_count}\n")
        print(f"Size of the Secound Speaker:{secound_speaker.sentences_count}\n")
        print(f"Size of the Other speakers:{other_speakers.sentences_count}\n")

        first_speaker,secound_speaker=down_sample(first_speaker,secound_speaker)
        
        first_speaker,other_spekaers=down_sample(first_speaker,other_speakers)

        if other_spekaers.sentences_count!=secound_speaker.sentences_count:
            secound_speaker=down_sample(secound_speaker,other_spekaers)

        print("*****************************************")
        print("size after down sampling:\n")
        print(f"Size of the First Speaker:{first_speaker.sentences_count}\n")
        print(f"Size of the Secound Speaker:{secound_speaker.sentences_count}\n")
        print(f"Size of the Other speakers:{other_speakers.sentences_count}\n")

        total_sentences=first_speaker.sentences+secound_speaker.sentences+other_spekaers.sentences
        binary_sentences=first_speaker.sentences+secound_speaker.sentences


        
        labels_binary=["first"]*first_speaker.sentences_count+["secound"]*secound_speaker.sentences_count
        labels_binary=np.array(labels_binary)
     

        labels_multi=(
            ["first"]*first_speaker.sentences_count+
            ["secound"]*secound_speaker.sentences_count+
            ["other"]*other_speakers.sentences_count


        )
        labels_multi=np.array(labels_multi)

    

        # section 3:3 - train the model
        knn_model,custom_knn_model,lr_model,custom_lr_model,vectorizer_tf=run_training(binary_sentences,labels_binary,'Binary')
        knn_model,custom_knn_model,lr_model,custom_lr_model,vectorizer_tf=run_training(total_sentences,labels_multi,'Multi')

       

        test_sentences=load_sentences(sentences_path)
        classify_speaker(test_sentences,'classification_results.txt',knn_model,vectorizer_tf,'bow')

        
        

              
   
    except Exception as e:
        raise e




# %%