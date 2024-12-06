#%%
import os
import re
import math
#import pandas as pd 
from docx import Document
from collections import defaultdict,Counter
import sys

import json




class Trigam_LM:

    def __init__(self,corpus):

       self.unigram_counts=Counter()
       self.bigram_counts=defaultdict(Counter)
       self.trigram_counts=defaultdict(Counter)
       self.total_toknes=0

       # define interpolation weights 
       self.lamda1,self.lamda2,self.lamda3= 0.5,0.3,0.2

       self._build_model(corpus)


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

            



       




    def calculate_prob_of_sentence(self,sentence):

        toknes= sentence.split()

        ## add dummy toknes to the sentence
        toknes.insert(0,"<s_0>")
        toknes.insert(1,"<s_1>")
        #toknes.append("<s_end>")

        log_prob=0.0
        size=len(self.unigram_counts)


        for i in range(2,len(toknes)):

            # get the current trigrams
            w1,w2,w3 =toknes[i-2] ,toknes[i-1] ,toknes[i]
            

            # calculate the trigram probability
            tri_count= self.trigram_counts[(w1,w2).get(w3,0)]
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
    
    def  generate_next_token(self,sentence):

        toknes=sentence.split()

        # toknes.insert(0,"<s_0>")
        # toknes.insert(1,"<s_1>")

        w1,w2=toknes[-2:]

        vocab_size=len(self.unigram_counts)
        probs=defaultdict(float)

        for token in self.unigram_counts:

             # calculate the trigram probability
            tri_count= self.trigram_counts[(w1,w2)][token]
            total_tri= sum(self.trigram_counts[(w1,w2)].values())
            tri_prob= (tri_count +1)/(total_tri+vocab_size)

            # calculate the bigram probability
            bi_count=self.bigram_counts[w2][token]
            total_bi= sum(self.bigram_counts[w2].values())
            bi_prob= (bi_count +1)/(total_bi+vocab_size)

            # calculate the unigram probability
            uni_count=self.unigram_counts[token]
            uni_prob= (uni_count+1)/(self.total_toknes + vocab_size)

            # apply linear interpolation 
            probs[token]= self.lamda1* tri_prob + self.lamda2*bi_prob + self.lamda3*uni_prob

            next_token=max(probs,key=probs.get)
            return next_token,probs[next_token]



















# %%
