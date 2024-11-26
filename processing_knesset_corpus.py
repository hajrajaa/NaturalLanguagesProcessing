#%%
import zipfile
import os
import re
#import pandas as pd 
#from docx import Document
#import docx
#import json


#%%

numbers={
    'אחת':1,
    'שתיים':2,
    'שלוש':3,
    'ארבע':4,
    'חמש':5,
    'שש':6,
    'שבע':7,
    'שמונה':8,
    'תשע':9,
    'עשר':10,

    'עשרים':20,
    'שלושים':30,
    'ארבעים':40,
    'חמשים':50,
    'ששים':60,
    'שבעים':70,
    'שמונים':80,
    'תשעים':90,

    'מאה':100,
    'מאתיים':200




}


#%% 
# Extract Zip file 
def extract_zip(path,detination):

    with zipfile.ZipFile(path,'r')as zip_ref:
        zip_ref.extractall(detination)



#%% classes 

class Sentence:

    def __init__(self):

        self.words = []

class Protocol:

    def __init__(self,knesset_num,protocol_type,file_name):

        self.knesset_num = knesset_num
        self.protocol_type= protocol_type
        self.file_name = file_name


        # add a list of Sentence objects to the Protocol object 
        ##self.sentences = []

        
#%%
#Extract data from protocol file names 
def extract_protocol_data(file):

   valid_format=re.match(r'(\d{2})_(ptm|ptv)_(.+).docx',file)

   if valid_format:
       
        knesset_num = int(valid_format.group(1))
        protocol_type= "plenary" if valid_format.group(2)=="ptm" else "committee"
        file_name=valid_format.group(3)
        protocol=Protocol(knesset_num,protocol_type,file_name)
        return protocol
   else:
         
        return None
   
# def extract_protocol_num(file):

#     try:

#         curr_doc=Document(file)


#         print (file)
#         return 1
    
    


#     except Exception as e :
#         return -1


def from_hebrew_to_number(text):

    # split the text by 's' or spaces
    parts=re.split(r'[-\s]',text)
    print(parts)

    sum=0
    prev_num=0

    for part in parts:
        
        if part.startswith(('ו','ה')):
            part=part[1:]

        if part=='מאות':
            sum-=prev_num
            sum+=prev_num*100
            prev_num=0
            continue

        if part=='עשרה':
            sum+=10
            prev_num=0
            continue

        if part in numbers:
            prev_num=numbers[part]
            sum+=prev_num
            
    return sum 




#%% Main code

if __name__ == "__main__":

    # zip_path="knesset_protocols.zip"

    # extract_to="knesset_protocols"

    # extract_zip(zip_path,extract_to)

    # protocol_files="knesset_protocols\protocol_for_hw1"

    text="חמש-מאות-ואחת-עשרה"
    print(from_hebrew_to_number(text))

    path=r"Knesset_protocols\protocol_for_hw1\15_ptv_490845.docx"

  


    # for file_name in os.listdir(protocol_files):

    #     protocol=extract_protocol_data(file_name)
    #     if protocol:
    #         print(protocol.knesset_num,protocol.protocol_type,protocol.file_name)
    #     else:
    #         print(f"invalid file name: {file_name}")




# %%
