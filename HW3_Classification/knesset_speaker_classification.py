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
from sklearn.preprocessing import LabelEncoder
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
        self.bow_vectors=None #to store the BoW vectors of the speaker's sentences
    
    def add_sentence(self,sentence):
        self.sentences.append(sentence)
        self.sentences_count+=1

    # ## SECTION 3 :1 - Extract BoW feature vector
    # def extract_BoW_vector(self , vectorizer=None):

    #     if not self.sentences:
    #         raise ValueError(f"No sentences for  speaker {self.name}")
        
    #     if not vectorizer:
    #         vectorizer=CountVectorizer()

    #     texts=[sentence['sentence_text'] for sentence in self.sentences if 'sentence_text' in sentence]
    #     self.bow_vectors=vectorizer.fit_transform(texts)

    #     return vectorizer




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
    knn=KNeighborsClassifier(n_neighbors=5)
    knn_cv_scores=cross_val_score(knn,features,labels,cv=cross_val,scoring='accuracy')
    # print("KNN Cross Validation Scores: ",knn_cv_scores)
    # print("KNN Average Cross Validation Score: ",np.mean(knn_cv_scores))

    #Logistic Regression model
    lr=LogisticRegression(max_iter=10000)
    lr_cv_scores=cross_val_score(lr,features,labels,cv=cross_val,scoring='accuracy')
    # print("Logistic Regression Cross Validation Scores: ",lr_cv_scores)
    # print("Logistic Regression Average Cross Validation Score: ",np.mean(lr_cv_scores))

    return knn_cv_scores.mean(),lr_cv_scores.mean()


# ## SECTION 3 :1 - Extract BoW feature vector
def extract_BoW_vector(vectorizer=None,sentences=None):

    if not sentences:
        raise ValueError(f"No sentences ")
    
    if not vectorizer:
        vectorizer=CountVectorizer()

    texts=[sentence['sentence_text'] for sentence in sentences if 'sentence_text' in sentence]
    bow_vectors=vectorizer.fit_transform(texts)

    return bow_vectors


    


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

        # print("First Speaker")
        # for sentence in first_speaker.sentences:
        #     print(sentence.get("sentence_text",None))
        # #     print("\n")
       
        # for sentence in other_spekaers.sentences:
        #     print(sentence.get("sentence_text",None))
        #     print("\n")

        total_sentences=first_speaker.sentences+secound_speaker.sentences+other_spekaers.sentences
        binary_sentences=first_speaker.sentences+secound_speaker.sentences
        
        
        #all_sentences=first_speaker.sentences+secound_speaker.sentences
        # section 3:1 _ extract the BoW feature vectors
        #vectorizer=CountVectorizer()
        vectorizer=TfidfVectorizer()
        bow_vector=extract_BoW_vector(vectorizer,total_sentences)
        binary_bow_vector=extract_BoW_vector(vectorizer,binary_sentences)
        # first_speaker.extract_BoW_vector(vectorizer)
        # secound_speaker.extract_BoW_vector(vectorizer)
        # bow_vectors=np.concatenate([first_speaker.bow_vectors.toarray(),secound_speaker.bow_vectors.toarray()],axis=0)
        # print(bow_vectors.shape)

        #print((first_speaker.bow_vectors.toarray()[1]))

        # section 3:2 - extract the features
        features_vector=extract_feature_vector(total_sentences)
        binary_features_vector=extract_feature_vector(binary_sentences)
        
        # first_speaker_features=extract_feature_vector(first_speaker.sentences)
        # secound_speaker_features=extract_feature_vector(secound_speaker.sentences)

        # section 3:3 - train the model
        
        first_speaker_df=pd.DataFrame({'sentence:':first_speaker.sentences,'label':first_speaker_name})
        secound_speaker_df=pd.DataFrame({'sentence:':secound_speaker.sentences,'label':secound_speaker_name})
        other_speakers_df=pd.DataFrame({'sentence:':other_spekaers.sentences,'label':"other"})
        combined_df=pd.concat([first_speaker_df,secound_speaker_df,other_speakers_df])
        combined_binary_df=pd.concat([first_speaker_df,secound_speaker_df])


        labels_binary=combined_binary_df['label'].values
        labels_binary=np.array(labels_binary)

        print("Binary Classification")
        bow_binary_knn,bow_binary_lr=train_model(binary_bow_vector,labels_binary)
        print("BoW Classifier Score -knn:  ",bow_binary_knn)
        print("BoW Classifier Score-lr: ",bow_binary_lr)
        feature_binary_knn,feature_binary_lr=train_model(binary_features_vector,labels_binary)
        print("Feature Classifier Score -knn: ",feature_binary_knn)
        print("Feature Classifier Score -lr: ",feature_binary_lr)


        

        

        labels=combined_df['label'].values

        labels=np.array(labels)

        #multi-class classification
        bow_classifier_knn,bow_classifier_lr=train_model(bow_vector,labels)
        print("BoW Classifier Score -knn:  ",bow_classifier_knn)
        print("BoW Classifier Score-lr: ",bow_classifier_lr)
        feature_classifier_knn,feature_classifier_lr=train_model(features_vector,labels)
        print("Feature Classifier Score -knn: ",feature_classifier_knn)
        print("Feature Classifier Score -lr: ",feature_classifier_lr)

        
        

              
   
    except Exception as e:
        raise e




# %%