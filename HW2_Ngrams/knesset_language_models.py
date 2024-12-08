#%%
import os
import re
import math
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
       self.lamda1,self.lamda2,self.lamda3= 0.5,0.3,0.2

       self._build_model(corpus)

       self.corpus=corpus


    def _build_model(self,corpus):

        for sentence in corpus:

            toknes=sentence.split()
            ## add dummy toknes to the sentence
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

        toknes= sentence.split()

        ## add dummy toknes to the sentence
        toknes.insert(0,"<s_0>")
        toknes.insert(1,"<s_1>")
        print (toknes)
        #toknes.append("<s_end>")

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

        w1,w2=sentence.split()[-2:]

        vocab_size=len(self.unigram_counts)
        probs=defaultdict(float)

        for token in self.unigram_counts:

            # skip the dummy toknes 
            if token in ["<s_0>","<s_1>"]:
                continue

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
#???????????????
def  get_k_n_t_collocation(self,k,n,t,metric="frequency"):

    try:
        ngram_counts=Counter()
        protocol_count=defaultdict(set)
        total_protocol=len(self.corpus)
        
        for protocol_name,sentences in self.corpus.items():

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
            print(committe_collocations_freq)
            for ngram in committe_collocations_freq:
                print(ngram[0])
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

       
        save_collocations(plenary_model,committee_model,output_file)
    
    except Exception as e:
        raise e






# %% Main 


### for check section one 
# corpus = [
#     "hello world",
#     "hello there",
#     "world of hello",
#     "there is a world"
# ]

# def test_calculate_prob_of_sentence():
#     lm = Trigram_LM(corpus)
#     sentence = "hello world"
#     prob = lm.calculate_prob_of_sentence(sentence)
#     print(f"Log probability of '{sentence}': {prob}")

# def test_generate_next_token():
#     lm = Trigram_LM(corpus)
#     context = "hello world"
#     next_token, prob = lm.generate_next_token(context)
#     print(f"Next token after '{context}': {next_token}, Probability: {prob}")

# test_calculate_prob_of_sentence()
# test_generate_next_token()

# %%