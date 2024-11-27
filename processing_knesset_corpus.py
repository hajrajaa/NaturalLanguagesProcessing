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
    'והתכנון','והתרבות','והמסחר'
    # ,
    # '','','',''
    # '','','','','','','','','','','','','','','','','',''
    }



postions_keywords={
    'סגן','סגנית','שר','שרת','מזכיר','מזכירת','השר','השרה','המשנה','היו"ר','תשובת','אורח','אורחת','דובר','דוברת','יור','פרופ','יו"ר','מר'
    'מ"מ','היו”ר',


}

skip={

    'קריאה','קריאות','נכחו','סדר היום','חברי הוועדה','חברי','מוזמנים','ייעוץ משפטי','מנהלת הוועדה','רישום פרלמנטרי',
    'משתתפים','נושא','רישום','מנהל/ת הוועדה','מנהל הוועדה','דיון'
     ,'כותבת',
    'פרלמנטארית','הצעות','הישבה','יום','הטקס','יועץ','קצרנית','מנחה','נוכחים','ברכות','הרצאה','סדר-היום','רשמה','הצגת'

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

        #self.protocol_num=extract_protocol_num(file_path)


        # add a list of Sentence objects to the Protocol object 
        ##self.sentences = []

        
#%% task 1.1
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
   
#%% task 1.2
   
def extract_protocol_num(file_path):

    # try:

      
    # except Exception as e:
    #     return -1 

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

#%% task 1.3

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


def get_speakrs_names(file_path):

    curr_doc=Document(file_path)

    speakers=[]

    for par in curr_doc.paragraphs:


        # remove white spaces
        match=re.match(r"^(.*?)\:",par.text.strip())

        if match and is_underlined(par):

            new_speaker=match.group(1).strip()
            new_speaker=filter_speakrs_names(new_speaker)
            

            
            
            if new_speaker not in speakers and new_speaker is not None:
                 speakers.append(new_speaker)

    #filterd_names=filter_speakrs_names(speakers)
    return speakers

def filter_speakrs_names(name):


    keywords=[]
    fileds=[]

    
    filtered_name=re.sub(r'\(.*?\)','',name).strip()

    ## check later if it is better to include it in the list instead 

    filtered_name= re.sub(r'<<.*?>>|<<.*?<<|>>.*?>>','',filtered_name)

    
    while filtered_name.startswith(('>','<')):
        filtered_name=filtered_name[1:]

                           

    if any( word in name for word in skip ) :
        return None 

    for keyword in postions_keywords:

        keyword=re.escape(keyword)
        keywords.append(keyword)

    
    keyword_pattern=r'\b(?:' +'|'.join(keywords)+r')\b'


    keyword_pattern=fr'{keyword_pattern}(?:\s*ל|\b[\s\-]*)?'


    filtered_name=re.sub(keyword_pattern,'',filtered_name)

    for field in postions_fields:
        field=re.escape(field)
        fileds.append(field)


    
    multy_feilds='|'.join("ל?"+ field +"(?:ו|ל|ול)?" for field in fileds)

    fields_pattern="\\b(?:"+multy_feilds+")\\b"

    filtered_name=re.sub(fields_pattern,'',filtered_name)
     
    ## remove () and << 
    #filtered_name=re.sub(r'\(.*?\)','',filtered_name).strip()


    return filtered_name.strip()




#%% Main code

if __name__ == "__main__":

    # zip_path="knesset_protocols.zip"

    # extract_to="knesset_protocols"

    # extract_zip(zip_path,extract_to)

    # protocol_files="knesset_protocols\protocol_for_hw1"

    text="חמש-מאות-ואחת-עשרה"
    print(from_hebrew_to_number(text))

    # path=r"Knesset_protocols\protocol_for_hw1\15_ptv_490845.docx"
    # path=r"Knesset_protocols\protocol_for_hw1"
    # file_name="15_ptv_490845.docx"
    # #print(extract_protocol_data(file_name))
    # #print(extract_protocol_num(path))
    # print(get_speakrs_names(os.path.join(path,file_name)))

    
    path=r"Knesset_protocols\protocol_for_hw1"
    for file_name in os.listdir(path):
        if file_name.endswith('.docx'):
            print(file_name)
            
            for speaker in get_speakrs_names(os.path.join(path,file_name)):
                print(speaker)
                print("\n")

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
