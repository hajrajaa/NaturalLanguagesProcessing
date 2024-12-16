#%%
import os
import math
import random
import pandas as pd 
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
       self.lamda1,self.lamda2,self.lamda3= 0.99959,0.0004,0.00001
       

       self._build_model(corpus)

       self.corpus=corpus


    def _build_model(self,corpus):

        for protocol_name,sentences in corpus.items():

            for sentence in sentences:

                 ## add dummy toknes to the sentence
                sentence="<s_0>"+" "+"s_1"+" "+sentence

                toknes=sentence.split()
               
               
                self.total_toknes+= len(toknes)

                for i in range(len(toknes)):

                    self.unigram_counts[toknes[i]]+=1

                    if i>0:
                        self.bigram_counts[toknes[i-1]][toknes[i]]+=1

                    if i>1:
                        self.trigram_counts[(toknes[i-2],toknes[i-1])][toknes[i]]+=1


    # section 1.1 
    def calculate_prob_of_sentence(self,sentence):


        
        toknes=sentence.split()

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
    
 


    def generate_next_token(self,sentence):

        #vocab_size=len(self.unigram_counts)
        probs=defaultdict(float)

        toknes=sentence.split()
        
      
        if len(toknes)<2:
            toknes=["<s_1>"]+toknes
            
        if len(toknes)<1 or toknes[0]!="<s_0>":
            
            toknes=["<s_0>"]+toknes

        w1,w2=toknes[-2],toknes[-1]

        if len(toknes)>1 : 

            for token in self.unigram_counts.keys():

                # skip the dummy toknes 
                if token in ["<s_0>","<s_1>"]:
                    continue

                curr_tigram=tuple((w1,w2,token))
               
                curr_sentence=curr_tigram[0]+" "+curr_tigram[1] + " "+curr_tigram[2]
            

                probs[token]=self.calculate_prob_of_sentence(curr_sentence)
              

        next_token=max(probs,key=probs.get)
       

        return next_token,probs[next_token]



# %% section 2.1 
# check if correct 
def  get_k_n_t_collocation(df_corpus,k,n,t,metric="frequency"):

    try:
        ngram_counts=Counter()   # total ngrams counter
        protocols_ngram=defaultdict(set)  # ngrams per protocol
        total_protocols=len(df_corpus['protocol_name'].unique())   # total number of protocols 
        curr_tf_protocol=[]  # to store TF per protocol
        #print(total_protocols)
        
        for _,row in df_corpus.iterrows():
            protocol_name=row['protocol_name']
            sentence=row['sentence_text']

            toknes=sentence.split()
            if len(toknes)<n:
                continue 

            # create ngrams
            ngrams=zip(*[toknes[i:] for i in range(n)])
            ngram_count_protocol=Counter()

            for ngram in ngrams:
                if any([token in ["<s_0>","<s_1>"] for token in ngram]):
                    continue
                
                ngram_counts[ngram]+=1
                protocols_ngram[ngram].add(protocol_name)
                ngram_count_protocol[ngram]+=1

            # compute the Tf for the current protocol
            curr_ngrams=sum(ngram_count_protocol.values())
            for ngram,count in ngram_count_protocol.items():
                curr_tf=count/curr_ngrams
                curr_tf_protocol.append(curr_tf)


        # filter ngrams - only ngrams with count >= t 
        filtered_ngrams={ngram : count for ngram,count in ngram_counts.items() if count>=t}
        

        if metric=="frequency":

            ranked_ngrams=sorted(filtered_ngrams.items(),key=lambda x:x[1],reverse=True)

        elif metric=="tfidf":
            tfidf={}
            
            for ngram in filtered_ngrams.items():

                # average TF across all protocols 
                tf=[protocol_tf.get(ngram,0) for protocol_tf in curr_tf_protocol]
                avg_tf=sum(tf)/len(tf)

                idf=math.log(total_protocols/ (len(protocols_ngram[ngram])+1))

                tfidf[ngram]= avg_tf*idf 
            
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


def save_collocations(df_plenary_corpus,df_committee_corpus,output_file):
    try:

        with open(output_file,'w',encoding='utf-8') as f_out: 

            ####################### section 2.2 : Two-gram collocations #######################
            f_out.write("Two-gram collocations:\n")
            f_out.write("Frequency:\n")

            f_out.write("Committee corpus:\n")
            committe_collocations_freq=get_k_n_t_collocation(df_committee_corpus,k=10,n=2,t=5,metric="frequency")
            #print(committe_collocations_freq)
            for ngram in committe_collocations_freq:
                #print(ngram[0])
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")
            

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_freq=get_k_n_t_collocation(df_plenary_corpus,k=10,n=2,t=5,metric="frequency")
            for ngram in plenary_collocations_freq:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            f_out.write("TF-IDF:\n")
            f_out.write("Committee corpus:\n")

            committe_collocations_tfidf=get_k_n_t_collocation(df_committee_corpus,k=10,n=2,t=5,metric="frequency")
            for ngram in committe_collocations_tfidf:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_tfidf=get_k_n_t_collocation(df_plenary_corpus,k=10,n=2,t=5,metric="frequency")
            for ngram in plenary_collocations_tfidf:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            ##################### section 2.3 : Three-gram collocations ###################### 
            f_out.write("Three-gram collocations:\n")
            f_out.write("Frequency:\n")

            f_out.write("Committee corpus:\n")
            committe_collocations_freq=get_k_n_t_collocation(df_committee_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in committe_collocations_freq:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_freq=get_k_n_t_collocation(df_plenary_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in plenary_collocations_freq:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            f_out.write("TF-IDF:\n")
            f_out.write("Committee corpus:\n")

            committe_collocations_tfidf=get_k_n_t_collocation(df_committee_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in committe_collocations_tfidf:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_tfidf=get_k_n_t_collocation(df_plenary_corpus,k=10,n=3,t=5,metric="frequency")
            for ngram in plenary_collocations_tfidf:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")


            ##################### section 2.4 : Four-gram collocations ########################

            f_out.write("Four-gram collocations:\n")
            f_out.write("Frequency:\n")

            f_out.write("Committee corpus:\n")
            committe_collocations_freq=get_k_n_t_collocation(df_committee_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in committe_collocations_freq:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_freq=get_k_n_t_collocation(df_plenary_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in plenary_collocations_freq:
                f_out.write(f"{ngram[0]}\n")
            f_out.write("\n")

            f_out.write("TF-IDF:\n")
            f_out.write("Committee corpus:\n")

            committe_collocations_tfidf=get_k_n_t_collocation(df_committee_corpus,k=10,n=4,t=5,metric="frequency")
            for ngram in committe_collocations_tfidf:
                f_out.write(f"{ngram[0]} \n")
            f_out.write("\n")

            f_out.write("Plenary corpus:\n") 
            plenary_collocations_tfidf=get_k_n_t_collocation(df_plenary_corpus,k=10,n=4,t=5,metric="frequency")
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
        mask_len= max(1,int(tokens_len* x / 100))

        valid_indices=[i for i in range(tokens_len) if toknes[i] not in ["<s_0>","<s_1>"]]

        if not valid_indices:
            masked_sentences.append(" ".join(toknes))
            continue


        mask_indices=random.sample(valid_indices,min(mask_len,len(valid_indices)))

        for i in mask_indices:
            toknes[i]='[*]'
        masked_sentences.append(" ".join(toknes))

    return masked_sentences 

## section 3.2 
def save_masked_sentences(corpus):

    orignal_output_file=os.path.join(output_dir,'original_sampled_sents.txt')
    masked_output_file=os.path.join(output_dir,'masked_sampled_sents.txt')
    masked_sentences=[]

    try:

        valid_sentences=[sentence for sentences in corpus.values() for sentence in sentences if len(sentence.split())>=5]

        sentences=random.sample(valid_sentences,10)

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


    predicted_toknes=[]

    curr_toknes=masked_sentence.split()
    predicted_toknes= curr_toknes.copy()

    masked_indices=[i for i in range(len(curr_toknes))if curr_toknes[i]=='[*]']
    

    for i in masked_indices:
            
            if i==0:
                curr_text=""
            else:
                curr_text=" ".join(predicted_toknes[:i])
                    
                
            next_token,prob=lm.generate_next_token(curr_text)
            predicted_toknes[i]=next_token
            
    predicted_sentence=[predicted_toknes[i] for i in masked_indices]

    return " ".join(predicted_toknes),predicted_sentence




 

      

def save_sampled_sentences(plenary_ml,committee_ml,output_file,orignal_sentences,masked_sentences):

    try:

        predicted_sentences=[]

        with open(output_file,'w',encoding='utf-8') as f_out:

            for i in range(len(orignal_sentences)):
                f_out.write("original_sentence: ")
                f_out.write(f"{orignal_sentences[i]}\n")
               
                f_out.write("masked_sentence: ")
                f_out.write(f"{masked_sentences[i]}\n")
                f_out.write("plenary_sentence: ")

                predicted_sentence,predicted_toknes=predict_masked_toknes(plenary_ml,masked_sentences[i])
                predicted_sentences.append(predicted_sentence)

                f_out.write(f"{predicted_sentence}\n")
                predicted_toknes=",".join(predicted_toknes)
                f_out.write(f"plenary_tokens: {predicted_toknes} \n")
        
                plenary_prob=plenary_ml.calculate_prob_of_sentence(predicted_sentence)
                committee_prob=committee_ml.calculate_prob_of_sentence(predicted_sentence)
                f_out.write(f"probability of plenary sentence in plenary corpus: {plenary_prob:.2f}\n")
                f_out.write(f"probability of plenary sentence in committee corpus: {committee_prob:.2f}\n")
        
        return predicted_sentences



    except Exception as e:
        raise e



def get_perplexity(lm,masked_sentences,predicted_sentences):

    total_perplexity=0.0
    sentences_num=0
    
    for predicted_sentence,masked_sentence in zip(predicted_sentences,masked_sentences):

        predicted_toknes=predicted_sentence.split()
        masked_toknes=masked_sentence.split()

        masked_indices=[i for i,curr_tokne in enumerate(masked_toknes)if curr_tokne=='[*]']
        

        if not masked_indices or len(masked_toknes)<3:
            continue

        sentence_log_prob=0.0
        sentence_tokne_count=0

        for i in masked_indices:

            if i<2:
                continue

            w1,w2,w3=masked_toknes[i-2],masked_toknes[i-1],predicted_toknes[i]

            trigram=" ".join([w1,w2,w3])

            curr_prob=lm.calculate_prob_of_sentence(trigram)
            sentence_log_prob+=curr_prob
            sentence_tokne_count+=1

            
        if sentence_tokne_count>0:
            avrg_log_prob=sentence_log_prob /sentence_tokne_count
            sent_perplexity=math.pow(2,-avrg_log_prob)
            total_perplexity+=sent_perplexity
            sentences_num+=1


    if sentences_num>0:

        average_perplexity=total_perplexity /sentences_num
    else:

        average_perplexity=float('inf')


    return average_perplexity




def save_perplexity(output_file,preplexity):

    try:

        with open(output_file,'w',encoding='utf-8') as f_out:

            f_out.write(f"{preplexity:.2f}")

    except Exception as e:
        raise e 


   



# %% Main 

if __name__=='__main__':

    try:
        
        if len(sys.argv)!=3:
            raise ValueError("Invalid input")
        
        input_file=sys.argv[1]
        output_dir=sys.argv[2]

        #ensure the output directory exists
        os.makedirs(output_dir,exist_ok=True)
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

        

        
        ## section 2   
        df_corpus=pd.DataFrame(corpus)
        df_plenary_corpus=df_corpus[df_corpus['protocol_type']=='plenary']
        df_committee_corpus=df_corpus[df_corpus['protocol_type']=='committee']
        collocations_output_file=os.path.join(output_dir,'knesset_collocations.txt')
        save_collocations(df_plenary_corpus,df_committee_corpus,collocations_output_file)


        ## section 3 
        orignal_sentences,masked_sentences=save_masked_sentences(committee_corpus)
  

        ## section 3.3 
        results_file=os.path.join(output_dir,'sampled_sents_results.txt')
        predicted_sentences= save_sampled_sentences(plenary_model,committee_model,results_file,orignal_sentences,masked_sentences)
    

        ## section 3.4 
        plenary_preplexity=get_perplexity(plenary_model,masked_sentences,predicted_sentences)
        perp_result_file=os.path.join(output_dir,"perplexity_result.txt")
        save_perplexity(perp_result_file,plenary_preplexity)

       
    
    except Exception as e:
        raise e
    
    







# %%