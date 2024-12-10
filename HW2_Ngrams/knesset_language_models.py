#%%
import os
import re
import math
import random
import pandas as pd 
from docx import Document
from collections import defaultdict,Counter
import sys

import json


class Trigram_LM:

    def __init__(self,corpus):

       self.unigram_counts=Counter()
       self.bigram_counts=defaultdict(Counter)
       self.trigram_counts=defaultdict(Counter)
       self.total_toknes=0

       # define interpolation weights 
       self.lamda1,self.lamda2,self.lamda3= 0.7,0.1,0.2

       self._build_model(corpus)

       self.corpus=corpus


    def _build_model(self,corpus):

        for protocol_name,sentences in corpus.items():

            for sentence in sentences:

                toknes=sentence.split()
                ## add dummy toknes to the sentence
                ## check if need to add space after each dummy toknes 
                toknes.insert(0,"<s_0>")
                toknes.insert(1,"<s_1>")

                self.total_toknes+= len(toknes)

                for i in range(len(toknes)):

                    self.unigram_counts[toknes[i]]+=1

                    if i>0:
                        self.bigram_counts[toknes[i-1]][toknes[i]]+=1

                    if i>1:
                        self.trigram_counts[(toknes[i-2],toknes[i-1])][toknes[i]]+=1


    # section 1.1 
    def calculate_prob_of_sentence(self,sentence):

        # checkkkk !!!!!
        toknes=sentence.split()

        ## add dummy toknes to the sentence
        toknes.insert(0,"<s_0>")
        toknes.insert(1,"<s_1>")
      

        log_prob=0.0
        size=len(self.unigram_counts)


        for i in range(2,len(toknes)):


            # get the current trigrams
            w1,w2,w3 =toknes[i-2] ,toknes[i-1] ,toknes[i]
            

            # calculate the trigram probability
            tri_count= self.trigram_counts[(w1,w2)].get(w3,0)
            total_tri= sum(self.trigram_counts[(w1,w2)].values())
            tri_prob= (tri_count +1)/(total_tri+size)

            # calculate the bigram probability
            bi_count=self.bigram_counts[w2].get(w3,0)
            total_bi= sum(self.bigram_counts[w2].values())
            bi_prob= (bi_count +1)/(total_bi+size)

            # calculate the unigram probability
            uni_count=self.unigram_counts.get(w3,0)
            uni_prob= (uni_count+1)/(self.total_toknes + size)

            # apply linear interpolation 
            total_prob= self.lamda1* tri_prob + self.lamda2*bi_prob + self.lamda3*uni_prob

            log_prob+= math.log(total_prob)
    
        return log_prob
    
    # section 1.2 
    def  generate_next_token(self,sentence):
        


        vocab_size=len(self.unigram_counts)
        probs=defaultdict(float)

        #w1,w2=sentence.split()[-2],sentence.split()[-1]
        toknes=sentence.split()
        
        if len(toknes)<2:
            toknes.insert(0,"<s_1>")
        if len(toknes)<1 or toknes[0]!="<s_0>":
            toknes.insert(0,"<s_0>")

        w1,w2=toknes[-2],toknes[-1]

        for token in self.unigram_counts:

            # skip the dummy toknes 
            if token in ["<s_0>","<s_1>"]:
                continue
            
            #print(w1,w2)

             # calculate the trigram probability
            tri_count= self.trigram_counts[(w1,w2)].get(token,0)
            
            total_tri= sum(self.trigram_counts[(w1,w2)].values())
            tri_prob= (tri_count +1)/(total_tri+vocab_size)

            # calculate the bigram probability
            bi_count=self.bigram_counts[w2].get(token,0)
            total_bi= sum(self.bigram_counts[w2].values())
            bi_prob= (bi_count +1)/(total_bi+vocab_size)

            # calculate the unigram probability
            uni_count=self.unigram_counts.get(token,0)
            uni_prob= (uni_count+1)/(self.total_toknes + vocab_size)

            # apply linear interpolation 
            probs[token]= self.lamda1* tri_prob + self.lamda2*bi_prob + self.lamda3*uni_prob

        next_token=max(probs,key=probs.get)
        return next_token,probs[next_token]



# %% section 2.1 
# check if correct 
def  get_k_n_t_collocation(corpus,k,n,t,metric="frequency"):

    try:
        ngram_counts=Counter()
        protocol_count=defaultdict(set)
        total_protocol=len(corpus)
        
        for protocol_name,sentences in corpus.items():

            for sentence in sentences:
                toknes=sentence.split()
                if len(toknes)<n:
                    continue 
                ngrams=zip(*[toknes[i:] for i in range(n)])

                for ngram in ngrams:
                    if any([token in ["<s_0>","<s_1>"] for token in ngram]):
                        continue
                    
                    ngram_counts[ngram]+=1
                    protocol_count[ngram].add(protocol_name)
            

        filtered_ngrams={ngram : count for ngram,count in ngram_counts.items() if count>=t}
        

        if metric=="frequency":

            ranked_ngrams=sorted(filtered_ngrams.items(),key=lambda x:x[1],reverse=True)

        elif metric=="tfidf":
            tfidf={}
            
            for ngram , f_t_d in filtered_ngrams.items():

                tf=f_t_d/sum(ngram_counts.values())

                idf=math.log(total_protocol/ len(protocol_count[ngram]))

                tfidf[ngram]= tf*idf 
            
            ranked_ngrams=sorted(tfidf.items(), key=lambda x:x[1],reverse=True)
        
        else :
            raise ValueError("Invalid index type")
        
        return ranked_ngrams[:k]

    except Exception as e:
        raise e
    
        




# %% 
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


def save_collocations(plenary_corpus,committee_corpus,output_file):
    try:

        with open(output_file,'w',encoding='utf-8') as f_out: 

            ####################### section 2.2 : Two-gram collocations #######################
            f_out.write("Two-gram collocations:\n")
            f_out.write("Frequency:\n")

            f_out.write("Committee corpus:\n")
            committe_collocations_freq=get_k_n_t_collocation(committee_corpus,k=10,n=2,t=5,metric="frequency")
            #print(committe_collocations_freq)
            for ngram in committe_collocations_freq:
                #print(ngram[0])
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")
            

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_freq=get_k_n_t_collocation(plenary_corpus,k=10,n=2,t=5,metric="frequency")
            for ngram in plenary_collocations_freq:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            f_out.write("TF-IDF:\n")
            f_out.write("Committee corpus:\n")

            committe_collocations_tfidf=get_k_n_t_collocation(committee_corpus,k=10,n=2,t=5,metric="frequency")
            for ngram in committe_collocations_tfidf:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_tfidf=get_k_n_t_collocation(plenary_corpus,k=10,n=2,t=5,metric="frequency")
            for ngram in plenary_collocations_tfidf:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            ##################### section 2.3 : Three-gram collocations ###################### 
            f_out.write("Three-gram collocations:\n")
            f_out.write("Frequency:\n")

            f_out.write("Committee corpus:\n")
            committe_collocations_freq=get_k_n_t_collocation(committee_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in committe_collocations_freq:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_freq=get_k_n_t_collocation(plenary_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in plenary_collocations_freq:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            f_out.write("TF-IDF:\n")
            f_out.write("Committee corpus:\n")

            committe_collocations_tfidf=get_k_n_t_collocation(committee_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in committe_collocations_tfidf:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_tfidf=get_k_n_t_collocation(plenary_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in plenary_collocations_tfidf:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")


            ##################### section 2.4 : Four-gram collocations ########################

            f_out.write("Four-gram collocations:\n")
            f_out.write("Frequency:\n")

            f_out.write("Committee corpus:\n")
            committe_collocations_freq=get_k_n_t_collocation(committee_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in committe_collocations_freq:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_freq=get_k_n_t_collocation(plenary_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in plenary_collocations_freq:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            f_out.write("TF-IDF:\n")
            f_out.write("Committee corpus:\n")

            committe_collocations_tfidf=get_k_n_t_collocation(committee_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in committe_collocations_tfidf:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_tfidf=get_k_n_t_collocation(plenary_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in plenary_collocations_tfidf:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

    except Exception as e:
        raise e
    

# %% section 3.1

def mask_toknes_in_sentences(sentences,x):

    # sentences : list of sentences
    # x : precentage of toknes to mask 

    masked_sentences=[]

    for sentence in sentences:

        toknes=sentence.split()
        tokens_len=len(toknes)
        maske_len= max(1,int(tokens_len* x / 100))


        mask_indices=random.sample(range(tokens_len),maske_len)

        for i in mask_indices:
            toknes[i]='[*]'
        masked_sentences.append(" ".join(toknes))

    return masked_sentences 

## section 3.2 
def save_masked_sentences(corpus):

    orignal_output_file='original_sampled_sents.txt'
    masked_output_file='masked_sampled_sents.txt'
    masked_sentences=[]

    try:

        all_sentences=[sentence for sentences in corpus.values() for sentence in sentences]

        sentences=random.sample(all_sentences,10)

        # section a
        with open(orignal_output_file,'w',encoding='utf-8') as f_out:
            for sentence in sentences:
                f_out.write(f"{sentence}\n")

        masked_sentences=mask_toknes_in_sentences(sentences,10)
       
        if not masked_sentences:
            raise ValueError("Empty masked sentenced")
        
        # section b 
        with open(masked_output_file,'w',encoding='utf-8') as f_out:
             for sentence in masked_sentences:
                 f_out.write(f"{sentence}\n")
        
        return sentences,masked_sentences


    except Exception as e:
        raise e 
    

## section 3.3 

def predict_masked_toknes(lm,masked_sentence):


    predicted_sentence=[]
    predicted_toknes=[]

    curr_toknes=masked_sentence.split()

    masked_indices=[i for i in range(len(curr_toknes))if curr_toknes[i]=='[*]']
    

    for i in masked_indices:
            #curr_text=" ".join(curr_toknes[max(0,i-2):i])
            #curr_text=" ".join(curr_toknes[:i])+" <MASK>"
            #print(curr_toknes[:i])
            curr_text=" ".join(curr_toknes[:i])
            #print(curr_text)
            next_token,prob=lm.generate_next_token(curr_text)
            curr_toknes[i]=next_token
            predicted_toknes.append(next_token)
        
    predicted_sentence.append(" ".join(curr_toknes))
    

    return " ".join(predicted_sentence),predicted_toknes




 

      

def save_sampled_sentences(plenary_ml,committee_ml,output_file,orignal_sentences,masked_sentences):

    try:

        with open(output_file,'w',encoding='utf-8') as f_out:

            for i in range(len(orignal_sentences)):
                f_out.write("original_sentence: ")
                f_out.write(f"{orignal_sentences[i]}\n")
                orignal_prob=plenary_ml.calculate_prob_of_sentence(orignal_sentences[i])
                print(orignal_prob)
                f_out.write("masked_sentence: ")
                f_out.write(f"{masked_sentences[i]}\n")
                f_out.write("plenary_sentence: ")

                predicted_sentence,predicted_toknes=predict_masked_toknes(plenary_ml,masked_sentences[i])

                f_out.write(f"{predicted_sentence}\n")
                predicted_toknes=",".join(predicted_toknes)
                f_out.write(f"plenary_tokens: {predicted_toknes} \n")
                

                ## correct this 
                # for token in predicted_toknes:
                #     f_out.write(f"{token}")
                #     if token!=predicted_toknes[-1]:
                #         f_out.write(",")
                #     else:
                #         f_out.write("\n")
                
                plenary_prob=plenary_ml.calculate_prob_of_sentence(predicted_sentence)
                committee_prob=committee_ml.calculate_prob_of_sentence(predicted_sentence)
                f_out.write(f"probability of plenary sentence in plenary corpus: {plenary_prob:.2f}\n")
                f_out.write(f"probability of plenary sentence in committee corpus: {committee_prob:.2f}\n")



    except Exception as e:
        raise e



# %% Main 

if __name__=='__main__':

    try:

        input_file='knesset_corpus.jsonl'
        corpus=load_corpus(input_file)

        plenary_corpus={}
        committee_corpus={}

        

        for data in corpus:

            protocol_type=data['protocol_type']
            protocol_name=data['protocol_name']
            sentence=data['sentence_text']

            if protocol_type=='plenary':

                if protocol_name  not in plenary_corpus:
                    plenary_corpus[protocol_name]=[]

                plenary_corpus[protocol_name].append(sentence)


            elif protocol_type=='committee':

                if protocol_name not in committee_corpus:

                    committee_corpus[protocol_name]=[]
                committee_corpus[protocol_name].append(sentence)
                    

        plenary_model=Trigram_LM(plenary_corpus)

        committee_model=Trigram_LM(committee_corpus)

        output_file='knesset_collocations.txt'

        
        ## section 2 
        #save_collocations(plenary_model.corpus,committee_model.corpus,output_file)

        ## section 3 
        orignal_sentences,masked_sentences=save_masked_sentences(committee_corpus)

        ## section 3.3 
        results_file='sampled_sents_results.txt'
        save_sampled_sentences(plenary_model,committee_model,results_file,orignal_sentences,masked_sentences)
    
    except Exception as e:
        raise e
    
    







# %%