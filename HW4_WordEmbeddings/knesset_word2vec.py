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
from sklearn.metrics.pairwise import cosine_similarity



# section 1:1 - Tokenization  (dont forget to rerwite the function )
def tokenize_sentences(sentences):

    tokenized_sentences = {}
    
    try:
        for sentence in sentences:
            original_sentence = sentence
            sentence = re.sub(r"[-_]", " ", sentence)
            curr_tokens = re.sub(r"[!().:;,?%!]", "", sentence).split()
            
            filtered_tokens = []
            for token in curr_tokens:
                token = token.strip()
                if len(token) < 2:
                    continue
                elif any(char.isdigit() for char in token):
                    continue
                filtered_tokens.append(token)

            tokenized_sentences[original_sentence] = filtered_tokens

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

# section 2:1 
def get_similar_words(words_list,word_vectors,output_dir='output'):

    similar_words={}
    #print(words_list)
    try:
        #output_path=os.path.join(output_dir,"knesset_similar_words.txt")
        with open("knesset_similar_words.txt",'w',encoding='utf-8') as file:
            for word in words_list:
                #print(word)
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

# section 2:2 - sentnce embeddings 
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



# section 2:3 - get similar sentences
def get_similar_sentences(choosen_sentneces,sentences_embeddings,word_vectors,toknes,output_dir):

    #output_path=os.path.join(output_dir,'knesset_similar_sentences.txt')
    try:
        tokenized_sentences=[tokenized for _,tokenized in choosen_sentneces]
        chosen_embeddings=get_sentence_embeddings(tokenized_sentences,word_vectors)
        similarty_matrix=cosine_similarity(chosen_embeddings,sentences_embeddings)

        with open('knesset_similar_sentences.txt','w',encoding='utf-8') as file:
         
            for i,(orignal_sentence,tokenized_sentence) in enumerate(choosen_sentneces):
                file.write(f"{orignal_sentence}: most similar sentences: ")

                most_similar_idx=similarty_matrix[i].argmax()
                most_similar_sentence=corpus_sentences[most_similar_idx]
                # check if we get to the last sentence
                if (i!=10):
                    file.write(f"{most_similar_sentence}\n")    
                else:
                    file.write(f"{most_similar_sentence}")  

    except Exception as e :
        raise e
      

############################################################################################
#section 4 - replace red words
def replace_red_words(word_vectors,red_words_mapping,output_dir):
    output_path=os.path.join(output_dir,'red_words_sentences.txt')
    os.makedirs(output_dir,exist_ok=True)
    replaced_words=[]
    try:
        with open('red_words_sentences.txt','w',encoding='utf-8') as file:

            for i,(red_words,sentence) in  enumerate(red_words_mapping.items()):
                updated_sentence=sentence
                curr_replacement=[]
                for red_word in red_words:
                    if red_word in word_vectors.key_to_index.keys():
                        similar_word=word_vectors.most_similar(red_word,topn=1)[0][0]
                        updated_sentence=updated_sentence.replace(red_word,similar_word)
                        curr_replacement.append((red_word,similar_word))
                file.write(f"{i+1}: ")
                file.write(f"{sentence}: {updated_sentence}\n")
                file.write(f"replaced words: ")
                for i,(red_word,similar_word) in  enumerate(curr_replacement):
                    file.write(f"({red_word},{similar_word})")
                    if i<len(curr_replacement)-1:
                        file.write(',')
                file.write('\n')
                

            replaced_words.extend(curr_replacement)


    except Exception as e :
        raise e
    
    return replaced_words
    





#%% main
if __name__=='__main__':
    print("aaaaaaaaaaa")
    try:
        
        input_path='knesset_corpus.jsonl'
        
        corpus=load_corpus(input_path) 
        
        corpus_sentences=[sentence.get('sentence_text','') for sentence in corpus]
        

        ###########################################################################################################################################
        # section 1:1
        toknes=tokenize_sentences(corpus_sentences)
        filtered_toknes=[]
        for sentence in toknes:
            filtered_toknes.append(toknes[sentence])

        
        

        # section 1:2 
        # build the word2vec model
        model=Word2Vec(sentences=filtered_toknes,vector_size=50,window=5,min_count=1)

        # save the trained model
        model.save('knesset_word2vec.model')

        #section 1:3 
        word_vectors=model.wv
        # print(word_vectors['ישראל'])   # get the vector of the word 
        # ##################################################################################################################################################
        # # section2 - similarity between words 
        # words_list=[
        #     'ישראל','גברת','ממשלה','חבר','בוקר','מים','אסור','רשות','זכויות'
        # ]
        # # to do : fix the path for the output 
        # ## section 2:1 - get the most 5 similar words for each word in the words list 
        # similar_words=get_similar_words(words_list,word_vectors)

        # ## section 2:2 sentence embeddings 
        # ## check what we should send as input to the function 
        # sentences_embeddings=get_sentence_embeddings(filtered_toknes,word_vectors)
       
        #print(sentences_embeddings[0])   


        # section 2:3 - get similar sentences
        ## to do : choose the sentnece not randomly !!!!
        # valid_sentences=[(orignal_sentences,toknes) for orignal_sentences,toknes in toknes.items() if len(toknes)>=4]
        # choosen_sentneces=random.sample(valid_sentences,min(10,len(valid_sentences)))

        
        # # to do : fix the path for the output 
        # get_similar_sentences(choosen_sentneces,sentences_embeddings,word_vectors,toknes,output_dir='output') 


        ## section 4 : replace red words 
        # sentences_with_red_words=[
        #     ".בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים",
        #     "בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים.",
        #     "בוקר טוב,אני פותח את הישיבה.",
        #     "שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום.",
        #     "אין מניעה להמשיך לעסוק בנושא."
        # ]

        red_words_mapping={
        ("דקות","הדיון"):".בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים",
        ("הוועדה","אני","ההסכם"):"בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים.",
        ("בוקר","פותח"):"בוקר טוב , אני פותח את הישיבה.",
        ("שלום","שמחים","היקר","קידום"):"שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום.",
        ("מניעה",):"אין מניעה להמשיך לעסוק בנושא."

        }

        x=replace_red_words(word_vectors,red_words_mapping,output_dir='output')
   
    except Exception as e:
        raise e
    
#%%