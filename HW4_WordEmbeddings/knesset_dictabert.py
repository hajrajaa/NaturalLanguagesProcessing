#%%
import os
import sys
import torch
import re
from transformers import AutoModelForMaskedLM, AutoTokenizer 


def load_sentences(file_path):
    sentences=[]
    try:
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                
                sentences.append(line.strip())
    except Exception as e:
        raise e
    
    return sentences



def predict_masked_toknes(tokenizer,model,masked_sentences):
    
    predictions=[]
    sampled_sentences=[]
    #inputs=[]
    try:
        for sentence in masked_sentences:
            toknezied=tokenizer(sentence,return_tensors='pt') 
           # inputs.append(toknezied)

            # get the model predictions
            with torch.no_grad():
                outpus=model(**toknezied)

            logits=outpus.logits
            mask_token_idx=torch.where(toknezied['input_ids']==tokenizer.mask_token_id)[1]

            sentence_predictions=[]
            for idx in mask_token_idx:
                token_logits=logits[0, idx, :]
                pred_token_id=torch.argmax(token_logits).item()
                pred_token=tokenizer.decode([pred_token_id])
                sentence_predictions.append(pred_token)

            predictions.append(sentence_predictions)
            sampled_sentence=sentence
            for pred_token in sentence_predictions:
                sampled_sentence=sampled_sentence.replace('[MASK]',pred_token,1)
            sampled_sentences.append(sampled_sentence)
        

    except Exception as e:
        raise e
    
    print(predictions)
    
    return sampled_sentences,predictions
    
def save_sampled_sentences(predictions,masked_sentences,sampled_sentences,output_dir):
        
        try:
            
            print("lolololo")
            #output_path=os.path.join(output_dir,'dictabert_results.txt')
            with open('dictabert_results.txt','w',encoding='utf-8') as f_out:
                for orignal_sentence,sampled_sentence,predictions in zip(masked_sentences,sampled_sentences,predictions):
                    f_out.write("masked_sentence: ")
                    f_out.write(f"{orignal_sentence}\n")

                    f_out.write("dictaBERT_sentence: ")
                    f_out.write(f"{sampled_sentence}\n")

                    predicted_toknes=",".join(predictions)
                    f_out.write(f"dictaBERT tokens: {predicted_toknes} \n")

                

        except Exception as e:
          raise e
        
        


if __name__=='__main__':
    try:

        input_path='masked_sampled_sents.txt'
        output_dir='HW4_WordEmbeddings'

        masked_sentences=load_sentences(input_path)
        masked_sentences=[sentence.replace('[*]','[MASK]') for sentence in masked_sentences]
      
        
        tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
        model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
        model.eval()
        sampled_sentences,predictions=predict_masked_toknes(tokenizer,model,masked_sentences)

        save_sampled_sentences(predictions,masked_sentences,sampled_sentences,output_dir)
        


       
    except Exception as e:
        raise e
    

#%%