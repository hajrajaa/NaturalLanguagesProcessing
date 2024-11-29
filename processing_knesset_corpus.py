#%%
import zipfile
import os
import re
import pandas as pd 
from docx import Document

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
    'חמישים':50,
    'שישים':60,
    'שבעים':70,
    'שמונים':80,
    'תשעים':90,

    'מאה':100,
    'מאתיים':200

}


postions_fields={ 

    'הכנסת','הכלכלה','התכנון','החינוך','התרבות','התעשייה','המסחר','האוצר','איכות הסביבה','הספורט','המשפטים','העבודה',
    'הרווחה','התחבורה','המשטרה','הבריאות','החקלאות',
    'התקשורת','ראש הממשלה','הפנים','קליטת העלייה','התיירות','ענייני דתות','התעסוקה','ביטחון פנים','התשתיות הלאומיות',
    'פיתוח הכפר','הבינוי','השיכון','המדע','הטכנולוגיה','ביטחון','החוץ','הבטיחות בדרכים',
    'הגנת הסביבה','קליטת העלייה','נושאים אסטרטגיים','ענייני מודיעין','אזרחים ותיקים','המודיעין','האנרגיה','המים','העלייה','הקליטה','השירותים החברתיים',
    'הביטחון','הפרלמנט האירופי','התשתיות','פנים','הלאומיות',''
    # ,
    # 'והספורט','הספורט','והתרבות ',''
    # '','','','','','','','','','','','','','','','','',''
    }



postions_keywords={
    'סגן','סגנית','שר','שרת','מזכיר','מזכירת','השר','השרה','המשנה','היו"ר','תשובת','אורח','אורחת','דובר','דוברת','יור','פרופ','יו"ר','מר'
    'מ"מ','היו”ר','במשרד','ד"ר','עו"ד','נצ"מ','ניצב','שופט','מל','נשיא','מ"מ','רשף','טפסר משנה','מר','פרופ\''



}

skip={

    'קריאה','קריאות','נכחו','סדר היום','חברי הוועדה','חברי','מוזמנים','ייעוץ משפטי','מנהלת הוועדה','רישום פרלמנטרי',
    'משתתפים','נושא','רישום','מנהל/ת הוועדה','מנהל הוועדה','דיון','יועצים','רכזת'
    ,'כותבת','הצעת','מנהלי הוועדה','הוועדה','נרשם ע"י','הצעת','הספרות','רשמת','רצח','החלטת','החלטה','יועצת','הישבה ננעלה'
    'פרלמנטארית','הצעות','הישיבה','יום','הטקס','יועץ','קצרנית','מנחה','נוכחים','ברכות','הרצאה','סדר-היום','רשמה','הצגת'

}   




# to convert hebrew numbers to numbers
def from_hebrew_to_number(text):

    # split the text by 's' or spaces
    parts=re.split(r'[-\s]',text)

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

# to check if the pargraph is underlined 
def is_underlined(paragraph):

    par_style=paragraph.style

    while par_style:
        if par_style and par_style.font.underline:
            return True

        par_style=par_style.base_style
    for run in paragraph.runs:
        if run.underline:
            return True
        
    # no underline found  
    return False



#%% classes 

class Sentence:

    def __init__(self,speaker,text):
        self.speaker=speaker
        self.text=text
    

class Protocol:

    def __init__(self,file_name,file_path):


        #data = Protocol.extract_protocol_data(file_name)

        # if data is None:
        #     raise ValueError('Invalid file name format')
        

        #self.knesset_num,self.protocol_type,self.file_name=data
    
        

        #self.protocol_num=Protocol.extract_protocol_num(file_path)

        self.sentences=self.extract_sentences(file_path)


        # #add a list of Sentence objects to the Protocol object 
        # self.sentences = Sentence(file_path)

        #self.speakers=Protocol.get_speakers_names(file_path)

    #task 1.1
    #Extract data from protocol file names
    @staticmethod
    def extract_protocol_data(file):

        valid_format=re.match(r'(\d{2})_(ptm|ptv)_(.+).docx',file)

        if valid_format:
            
                knesset_num = int(valid_format.group(1))
                protocol_type= "plenary" if valid_format.group(2)=="ptm" else "committee"
                file_name=valid_format.group(3)
                return knesset_num,protocol_type,file_name
        else:
                
                return None
        

    
    
    # task 1.2
    @staticmethod 
    def extract_protocol_num(file_path):

        try:

            curr_doc=Document(file_path)
            

            patterns=[
                r"פרוטוקול מס['\"]?\s*(\d+)",
                r"ישיבה מס['\"]?\s*(\d+)",
                r"הישיבה\s+([\w-]+)\s+של"
            ]

            for par in curr_doc.paragraphs:

                for pattern in patterns:
                    
                    match=re.search(pattern,par.text)
                    
                    if match:
                        
                        if pattern==patterns[2]:
                            return from_hebrew_to_number(match.group(1))
                        else:
                            return int(match.group(1))
                        
            # no match found            
            return -1 
        
        except Exception as e:
            return -1 
        

    
    def extract_sentences(self,file_path):

        curr_doc=Document(file_path)

        sentences=[]


        for par_index,par in enumerate (curr_doc.paragraphs):

            if is_underlined(par):
                
                
                # remove white spaces
                match=re.match(r"^(.*?)\:",par.text.strip())

                if match :
                    
                    new_speaker=match.group(1).strip()

                    if any( word in new_speaker for word in skip ) or any(new_speaker.startswith(word)for word in skip) :
                        continue  

                    text=self.extract_speaker_text(par_index+1,curr_doc)
                    print(text)
                    new_speaker=self.filter_speakrs_names(new_speaker)
                    for sentence in sentences:
                        if sentence.speaker==new_speaker:
                             sentence.text+="\n"+ text
                             break
                   
                    new_sentence=Sentence(new_speaker,text)
                    sentences.append(new_sentence)

                  
        return sentences
    
    @staticmethod
    def extract_speaker_text(par_index,documnet):

        speaker_text="" 

        for par in documnet.paragraphs[par_index:]:

            if is_underlined(par):
                break
            speaker_text+= par.text.strip() +"\n"

        return speaker_text
    
    @staticmethod
    def filter_speakrs_names(name):


        keywords=[]
        fileds=[]

        
        filtered_name=re.sub(r'\(.*?\)','',name).strip()


        ## check later if it is better to include it in the list instead 

        filtered_name= re.sub(r'<<.*?>>|<<.*?<<|>>.*?>>','',filtered_name)

        
        while filtered_name.startswith(('>','<')):
            filtered_name=filtered_name[1:]



        for keyword in postions_keywords:

            keyword=re.escape(keyword)
            keywords.append(keyword)

        for field in postions_fields:
            field=re.escape(field)
            fileds.append(field)

        
        keyword_pattern=r'\b(?:' +'|'.join(keywords)+r')\b'
        keyword_pattern=fr'{keyword_pattern}(?:\s*ל|\b[\s\-]*)?'
        


        multy_feilds='|'.join(f"(?:ו?ל?)?{field}(?:ול|,|ל|ו)?" for field in fileds)

        fields_pattern=r'\b(?:'+ multy_feilds +r')\b'

        prev_filtered_name=None
        while filtered_name!=prev_filtered_name:
            prev_filtered_name=filtered_name
            filtered_name=re.sub(keyword_pattern,'',filtered_name)
            filtered_name=re.sub(fields_pattern,'',filtered_name)
            filtered_name=re.sub(r'\s*,\s*','',filtered_name)
            

            
        # remove extra white spaces
        filtered_name=re.sub(r'\s{2,}','',filtered_name)


        return filtered_name.strip()
                       
                      




#%% Main code

if __name__ == "__main__":
    
    path=r"Knesset_protocols\protocol_for_hw1"
    for file_name in os.listdir(path):
        if file_name.endswith('.docx'):
            print(file_name)
            protocol=Protocol(file_name,os.path.join(path,file_name))
            #print(protocol.knesset_num,protocol.protocol_type,protocol.file_name)
            
            for sentence in protocol.sentences:
                if sentence in None:
                    print("looooooooooooo")
                print("speaker:",sentence.speaker)
                print("text:",sentence.text)
                

    # speaker='סגן שרת החינוך והתרבות וליד גולדמן'
    # speaker1=' סגן שר החינוך , התרבות והספורט'

    # print(filter_speakrs_names(speaker))
    # print(filter_speakrs_names(speaker1))

    # speaker=['השר לקליטת העלייה','שר הכלכלה','סגן השר לביטחון','סגנית מזכיר הכנסת']

    # filtered_speaker=filter_speakrs_names(speaker)
    # print(filtered_speaker)


    # for file_name in os.listdir(path):
    #      if file_name.endswith('.docx'):        
    #          print(extract_protocol_num(os.path.join(path,file_name)))
    
    

  


    # for file_name in os.listdir(protocol_files):

    #     protocol=extract_protocol_data(file_name)
    #     if protocol:
    #         print(protocol.knesset_num,protocol.protocol_type,protocol.file_name)
    #     else:
    #         print(f"invalid file name: {file_name}")




# %%
