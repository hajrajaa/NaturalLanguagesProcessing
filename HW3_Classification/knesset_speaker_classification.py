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
###########################################################################################################################################

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

def load_sentences(file_path):
    sentences=[]
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                sentences.append(line.strip())
    except Exception as e:
        raise e
    
    return sentences


##########################################################################################################################################
## scection 1
#  get the most common speakers
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


# split the data to the speakers classes 
def split_data(first_speaker,secound_speaker,other_speakers,corpus):

    # aliased for the speakers name 
    aliases={
        first_speaker.name:{"ר' ריבלין", "רובי ריבלין"},
             secound_speaker.name: {"אברהם בורג"}
    }

    for data in corpus:
        
        curr_name=data.get("speaker_name",None)

    
        if curr_name in aliases.get(first_speaker.name,set()) or curr_name==first_speaker.name:
            first_speaker.add_sentence(data)

        elif curr_name in aliases.get(secound_speaker.name,set()) or curr_name==secound_speaker.name :
            secound_speaker.add_sentence(data)
      
        else:
            other_speakers.add_sentence(data)

    return first_speaker,secound_speaker,other_speakers
##############################################################################################################################################################
## section 2
# down sample the data 
def down_sample(first_speaker,secound_speaker):

    first_speaker_size=first_speaker.sentences_count
    secound_speaker_size=secound_speaker.sentences_count


    if first_speaker_size>secound_speaker_size:
        first_speaker.sentences=random.sample(first_speaker.sentences,secound_speaker_size)
        first_speaker.sentences_count=secound_speaker_size
    elif first_speaker_size<secound_speaker_size:
        secound_speaker.sentences=random.sample(secound_speaker.sentences,first_speaker_size)
        secound_speaker.sentences_count=first_speaker_size

    return first_speaker,secound_speaker


def run_down_sampling(first_speaker,secound_speaker,other_speakers):
        
        # print("DOWN SAMPLING THE DATA:\n")
        # print("size before down sampling:\n")
        # print(f"Size of the First Speaker:{first_speaker.sentences_count}\n")
        # print(f"Size of the Secound Speaker:{secound_speaker.sentences_count}\n")
        # print(f"Size of the Other speakers:{other_speakers.sentences_count}\n")

        first_speaker,secound_speaker=down_sample(first_speaker,secound_speaker)
        
        first_speaker,other_spekaers=down_sample(first_speaker,other_speakers)

        if other_spekaers.sentences_count!=secound_speaker.sentences_count:
            secound_speaker=down_sample(secound_speaker,other_spekaers)

        # print("*****************************************")
        # print("size after down sampling:\n")
        # print(f"Size of the First Speaker:{first_speaker.sentences_count}\n")
        # print(f"Size of the Secound Speaker:{secound_speaker.sentences_count}\n")
        # print(f"Size of the Other speakers:{other_speakers.sentences_count}\n")

        return first_speaker,secound_speaker,other_speakers

######################################################################################################################################################

### SECTION 3 
# Section 3:1 - Extract BoW feature vector
def extract_BoW_vector(vectorizer=None,sentences=None):

    if not sentences:
        raise ValueError(f"No sentences ")
    
    bow_vectors=vectorizer.fit_transform(sentences)

    return bow_vectors


def get_unique_words(first_speaker,secound_speaker,other_speakers):
                     
    first_words_count=Counter()
    secound_words_count=Counter()
    other_words_count=Counter()

    for sentence in first_speaker.sentences:
        words=sentence.get('sentence_text','').split()
        first_words_count.update(words)

    for sentence in secound_speaker.sentences:
        words=sentence.get('sentence_text','').split()
        secound_words_count.update(words)

    for sentence in other_speakers.sentences:
        words=sentence.get('sentence_text','').split()
        other_words_count.update(words)

    unique_words=set()
    for word,count in first_words_count.items():
        if count >= 250 and count >= 2*max(secound_words_count[word],other_words_count[word]):
            unique_words.add(word)

    for word,count in secound_words_count.items():
        if count >= 250 and count >= 2*max(first_words_count[word],other_words_count[word]):
            unique_words.add(word)

    for word,count in other_words_count.items():
        if count >= 250 and count >=2* max(first_words_count[word],secound_words_count[word]):
            unique_words.add(word)

    to_exlude=['\'','"','!','(',')','.',',',';','?','-','%',':','–']
    unique_words=[word for word in unique_words if word not in to_exlude]
    
    return unique_words 
    

# section 3:2 - Extract feature vector 
def extract_feature_vector(sentences):

    global special_words

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
        for word in special_words:
                 feature_vector[f'special_word_{word}']=curr_sentence_txt.count(word)
    

        features.append(feature_vector)


    return pd.DataFrame(features)


#############################################################################################################################
# Section 4 
# train the models 
def train_model(features,labels):
    
    cross_val=StratifiedKFold(n_splits=5, shuffle=True,random_state=42)

    #KNN model
    knn_model=KNeighborsClassifier(n_neighbors=5)
    knn_cv_scores=cross_val_score(knn_model,features,labels,cv=cross_val,scoring='accuracy')
    knn_model.fit(features,labels)


    #Logistic Regression model
    lr_model=LogisticRegression(max_iter=10000)
    lr_cv_scores=cross_val_score(lr_model,features,labels,cv=cross_val,scoring='accuracy')
    lr_model.fit(features,labels)
   

    return knn_model,knn_cv_scores,lr_model,lr_cv_scores

def run_training(sentences,labels,classification_type='Multi-Class') :
    try:

        
        sentences_txt=[sentence.get('sentence_text','') for sentence in sentences]

        vectorizer_count=CountVectorizer()
        bow_count_vector=extract_BoW_vector(vectorizer_count,sentences_txt)
        
        
        vectorizer_tf=TfidfVectorizer()
        bow_vector=extract_BoW_vector(vectorizer_tf,sentences_txt)
        features_vector=extract_feature_vector(sentences)

        
    
        #print(f"{classification_type} Classification:")
        combind_report={}


        #print("CountVectorizer:\n")
        count_knn_model,bow_count_knn,count_lr_model,bow_count_lr=train_model(bow_count_vector,labels)
        combind_report['CountVectorizer_KNN']=classification_report(labels,count_knn_model.predict(bow_count_vector), output_dict=True)
        combind_report['CountVectorizer_LR']=classification_report(labels,count_lr_model.predict(bow_count_vector), output_dict=True)
        # print("BoW classifier score -KNN:\n",bow_count_knn)
        # print("BoW classifier score mean-KNN:",bow_count_knn.mean())
        # print("BoW classifier score -LR:\n",bow_count_lr)
        # print("BoW classifier score mean-LR:",bow_count_lr.mean())

        # print("******************************************************************")
        # print("TfidfVectorizer:\n")
        knn_model,bow_knn,lr_model,bow_lr=train_model(bow_vector,labels)
        combind_report['TfidfVectorizer_KNN']=classification_report(labels,knn_model.predict(bow_vector), output_dict=True)
        combind_report['TfidfVectorizer_LR']=classification_report(labels,lr_model.predict(bow_vector), output_dict=True)
        # print("BoW Classifier Score -KNN:\n",bow_knn)
        # print("BoW classifier score mean-KNN:",bow_knn.mean())
        # print("BoW Classifier Score-LR:\n",bow_lr)
        # print("BoW classifier score mean-LR:",bow_lr.mean())

        # print("******************************************************************")
        # print("Custom Feature Vector:\n")
        custom_knn_model,feature_knn,custom_lr_model,feature_lr=train_model(features_vector,labels)
        combind_report['CustomFeatureVector_KNN']=classification_report(labels,custom_knn_model.predict(features_vector), output_dict=True)
        combind_report['CustomFeatureVector_KNN_LR']=classification_report(labels,custom_lr_model.predict(features_vector), output_dict=True)
        # print("Custom Feature Classifier Score -KNN:\n",feature_knn)
        # print("Custom Feature Classifier Score mean -KNN: ",feature_knn.mean())
        # print("Custom Feature Classifier Score -LR: \n",feature_lr)
        # print("Custom Feature Classifier Score mean -LR: ",feature_lr.mean())

        # print("\nClassification Report:\n")
        # for model,report in combind_report.items():
        #     print(f"Model:{model}")
        #     print("\n")
        #     print(pd.DataFrame(report).transpose())
        #     print("\n")
    

        return knn_model,custom_knn_model,lr_model,custom_lr_model,vectorizer_tf

    except Exception as e:
        raise e
    
#############################################################################################################################
# Section 5 - classify the speakers 
def classify_speaker(sentences,output_fir,model,vectorizer=None):

    try:
        output_path=os.path.join(output_dir,'classification_results.txt')
        with open(output_path,'w',encoding='utf-8')as f:

            sentences_features=vectorizer.transform(sentences)
            
            predictions=model.predict(sentences_features)
           
            for pred in predictions:
                f.write(f"{pred}\n")


    except Exception as e:
        raise e
#############################################################################################################################

def run(input_path,sentences_texts_file,output_dir):
     
    # load the corpus
    corpus=load_corpus(input_path) 

    # get the most common speakers 
    first_speaker_name,secound_speaker_name=get_common_speakers(corpus)

    # section 1 - split the data 
    first_speaker,secound_speaker,other_speakers=split_data(first_speaker,secound_speaker,other_speakers,corpus)

    total_sentences=first_speaker.sentences+secound_speaker.sentences+other_speakers.sentences
    binary_sentences=first_speaker.sentences+secound_speaker.sentences

    # get the unique words 
    special_words= get_unique_words(first_speaker,secound_speaker,other_speakers)
    # print(f"Number of unique words: {len(special_words)}")
    # print(special_words)

    # labels for the binary and multu-class classification 
    labels_binary=["first"]*first_speaker.sentences_count+["secound"]*secound_speaker.sentences_count
    labels_binary=np.array(labels_binary)
    

    labels_multi=(
        ["first"]*first_speaker.sentences_count+
        ["secound"]*secound_speaker.sentences_count+
        ["other"]*other_speakers.sentences_count


    )
    labels_multi=np.array(labels_multi)


    # section 4 - train the model
    knn_model,custom_knn_model,lr_model,custom_lr_model,vectorizer_tf=run_training(binary_sentences,labels_binary,'Binary')
    knn_model,custom_knn_model,lr_model,custom_lr_model,vectorizer_tf=run_training(total_sentences,labels_multi,'Multi-Class')

    
    # section 5 - classify the speaker 
    test_sentences=load_sentences(sentences_texts_file)
    #classify_speaker(test_sentences,'classification_results_lr.txt',lr_model,vectorizer_tf)
    classify_speaker(test_sentences,output_dir,knn_model,vectorizer_tf)



# %% Main 

if __name__=='__main__':

    try:
        if len(sys.argv)!=4:
            raise ValueError("Invalid Input Arguments!!")
        
        input_path=sys.argv[1]

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File Not Found:{input_path}")
        
        sentences_texts_file=sys.argv[2]

        if not os.path.exists(sentences_texts_file):
            raise FileNotFoundError(f"File Not Found:{sentences_texts_file}")

        output_dir=sys.argv[3]

        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"File Not Found:{output_dir}")
        
        run(input_path,sentences_texts_file,output_dir)
        
        
   
    except Exception as e:
        raise e




# %%