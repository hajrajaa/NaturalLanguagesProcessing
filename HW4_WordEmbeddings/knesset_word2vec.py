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

## SECTION 1 FUNCTIONS
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

# section 1:1 - Tokenization
def tokenize_sentences(sentences):
    ## FIX IT 
    tokenized_sentences={}

    try:
        for sentence in sentences:
            original_sentence=sentence
            sentence=re.sub(r"[-_]"," ",sentence)
            curr_toknes=re.sub(r"[!().:;,?%]","",sentence).split(' ')


            filtered_toknes=[]
            for token in curr_toknes:
                token=token.strip()
                if len(token)<2:
                    continue
                elif any(char.isdigit() for char in token):
                    continue
                filtered_toknes.append(token)
            tokenized_sentences[original_sentence]=filtered_toknes

    except Exception as e:
        raise e
    return tokenized_sentences

def build_word2Vec_model(input_path,output_dir):

    os.makedirs(output_dir,exist_ok=True)
    model_path=os.path.join(output_dir,'knesset_word2vec.model')
    try:
        corpus=load_corpus(input_path)

        corpus_sentences=[sentence.get('sentence_text','') for sentence in corpus]

        toknes=tokenize_sentences(corpus_sentences)
        
        
        filtered_toknes=[]
        for sentence in toknes:
            filtered_toknes.append(toknes[sentence])
    
        if os.path.exists(model_path):
            model=Word2Vec.load(model_path)
        

        else:
             # build the word2vec model
             model=Word2Vec(sentences=filtered_toknes,vector_size=50,window=5,min_count=3)

             # save the trained model
             model.save(model_path)


        word_vectors=model.wv
        #print(word_vectors['ישראל'])   # get the vector of the word 

        return model,word_vectors,corpus_sentences,filtered_toknes

    except Exception as e :
        raise e



#####################################################################################################################################################################


# section 2:1 
def get_similar_words(words_list,word_vectors,output_dir):

    ## fix the path for the output 

    similar_words={}
    try:
        #output_path=os.path.join(output_dir,"knesset_similar_words.txt")
        with open("knesset_similar_words.txt",'w',encoding='utf-8') as file:
            for word in words_list:
                
                similarity_scores={other_word: word_vectors.similarity(word,other_word) for other_word in word_vectors.key_to_index if other_word!=word }
                
                sorted_similarty_scores=sorted(similarity_scores.items(),key=lambda item: item[1], reverse=True)
                file.write(f"{word}: ")
                for w,score in sorted_similarty_scores[:4]:
                    file.write(f"({w},{score}),")
                last_word,last_score=sorted_similarty_scores[4]
                file.write(f"({last_word},{last_score})")
                if word!=words_list[-1]:
                    file.write('\n')

    except Exception as e :
        raise e
    
    return similar_words


# section 2:2 - sentnce embeddings 
def get_sentence_embeddings(sentences,word_vectors):

    sentences_embeddings=[]

    for sentence in sentences:
        
        sentence_vector=np.zeros(word_vectors.vector_size)
        words_num=0   # number of words in the sentence 
        for token in sentence:
            # to ensure that the token is in the word vectors
            if token in word_vectors.key_to_index.keys():
                sentence_vector+=word_vectors[token]
                words_num+=1

            if words_num>0 :
                sentence_vector/=words_num
        sentences_embeddings.append(sentence_vector)  

    return sentences_embeddings              



# section 2:3 - get similar sentences
def get_similar_sentences(choosen_sentneces,sentences_embeddings,word_vectors,output_dir):

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
      


##################################################################################################################################################    
## section 4 - replace red words
def  replace_red_words(word_vectors,red_words_mapping,output_dir='output'):

    file_path = os.path.join(output_dir, "red_words_sentences.txt")

    positive_words = {
        'דקות': ['דקות','זמן','אחד'], 'הדיון': ['הדיון','שיחה'], 'הוועדה': ['הוועדה'], 'אני': ['אני'],
        'ההסכם': ['ההסכם','הוויכוח'], 'בוקר': ['ביום','שבוע','חמישי'], 'פותח': ['מתחיל','מסיים'], 'שלום': ['סליחה','הנכבדים'],
        'שמחים': ['שמחים','מודיעים'], 'היקר': ['גבר','תיאור'], 'קידום': ['רמה','קידום'], 'מניעה': ['מניעה','מתנגד']
    }
    negative_words = {
        
        'היקר': ['מחיר'], 'בוקר': ['יומיים','רביעי'], 'קידום': ['פריפריה']
    }
    chosen_replacements_index = {
        'דקות': 2, 'הדיון': 0, 'הוועדה': 0, 'אני': 1,
        'ההסכם': 2, 'בוקר': 0, 'פותח': 0, 'שלום': 2,
        'שמחים': 1, 'היקר': 2, 'קידום': 0, 'מניעה': 0
    }
    new_sentences = []
    with open(file_path, 'w', encoding='utf-8') as file:
        
        for i,(red_words,sentence) in  enumerate(red_words_mapping.items()):
            
            curr_replacement=[]
            new_sentence = sentence
            for word in red_words:
                
                
                if word in negative_words:
                    similar_word = word_vectors.most_similar(positive=positive_words[word], negative=negative_words[word],topn=3)
                    
                else:
                    similar_word = word_vectors.most_similar(positive=positive_words[word], topn=3)

                print(word,similar_word)
                
                replacing_word = similar_word[chosen_replacements_index[word]][0]
                curr_replacement.append((word,replacing_word))
                new_sentence = new_sentence.replace(word, replacing_word)

            new_sentences.append(new_sentence)  
            file.write(f"{i+1}: ")
            file.write(f"{sentence}: {new_sentence}\n")
            file.write(f"replaced words: ")
            for i,(red_word,replaced_word) in  enumerate(curr_replacement):
                file.write(f"({red_word},{replaced_word})")
                if i<len(curr_replacement)-1:
                    file.write(',')
            file.write('\n')  



        

#%% main
if __name__=='__main__':
    
    try:
        
        input_path='knesset_corpus.jsonl'

        ###########################################################################################################################################
        ## SECTION 1 
        model,word_vectors,corpus_sentences,filtered_toknes=build_word2Vec_model(input_path,'output')
        
        ###################################################################################################################################################
        ## SECTION 2 - SIMILARITY BETWEEN WORDS  AND SENTENCES
        words_list=[
            'ישראל','גברת','ממשלה','חבר','בוקר','מים','אסור','רשות','זכויות'
        ]
        # to do : fix the path for the output 
        ## section 2:1 - get the most 5 similar words for each word in the words list 
        similar_words=get_similar_words(words_list,word_vectors,output_dir='output')

        ## section 2:2 sentence embeddings  
        sentences_embeddings=get_sentence_embeddings(filtered_toknes,word_vectors)

     

        #section 2:3 - get similar sentences
        # to do : choose the sentnece not randomly !!!!
        choosen_sentneces=[
            " לימוד השפה העברית הוא חובה בבתי-הספר הערביים, אבל עד היום לא הפך לימוד השפה הערבית חובה בבתי-הספר היהודיים.",
            "ועדת החינוך והתרבות תדון בהצעת חוק הארכיונים (תיקון – הוראות שונות), התשנ\"ה–1995.",
            "מטרת חוק הדרכים היא להסדיר את השילוט שמוצב לצדי הדרכים.",
            "חבר הכנסת לס, אני קוראת אותך לסדר בפעם הראשונה.",
            "בישיבה הקודמת ביקשנו את חוות הדעת של היועץ המשפטי לממשלה וגם היועצת המשפטית לוועדה היתה אמורה לחוות דעתה.",
            "תודה ליושב-ראש האופוזיציה, ראש הממשלה לשעבר, חבר הכנסת שמעון פרס.",
            "כל התיקון הזה הוא לגבי בנייה חדשה או על תוספת בנייה במבנים ישנים.",
            "ביצוע ניסוי לאכיפה אלקטרונית אוטומטית של מהירות ובעבירות חמורות אחרות בשני קטעי כביש בין-עירוניים, כשלב מקדים לקראת הקמת מערכת חדשנית בחלק ניכר מהכבישים הבין-עירוניים בארץ.",
            " אשפוז ביתי יכול לחסוך הרבה מאוד חולים \"מיותרים\" בבתי החולים, שאפשר יהיה להגיע אליהם.",
            " הבוקר, לפני שהגעתי לכאן, ראיתי בישיבה מיוחדת עם מפכ\"ל המשטרה את סרטי הווידיאו – מה שהיה בבית-המשפט."

        ]
        choosen_sentences_toknes=tokenize_sentences(choosen_sentneces)
        valid_sentences=[(choosen_sentneces,choosen_sentences_toknes) for choosen_sentneces,choosen_sentences_toknes in choosen_sentences_toknes.items() if choosen_sentences_toknes]
        
        
        # to do : fix the path for the output 
        get_similar_sentences(valid_sentences,sentences_embeddings,word_vectors,output_dir='output') 

        ###################################################################################################################################################

        # section 4 : replace red words 
       
        red_words_mapping={
        ("דקות","הדיון"):".בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים",
        ("הוועדה","אני","ההסכם"):"בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים.",
        ("בוקר","פותח"):"בוקר טוב , אני פותח את הישיבה.",
        ("שלום","שמחים","היקר","קידום"):"שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום.",
        ("מניעה",):"אין מניעה להמשיך לעסוק בנושא."

        }

        replace_red_words(word_vectors,red_words_mapping,output_dir='output')
   
    except Exception as e:
        raise e
    
#%%