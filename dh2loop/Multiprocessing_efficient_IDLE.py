import psycopg2
from datetime import datetime
import pandas as pd
import multiprocessing as mp
import time
from multiprocessing import Pool
import psycopg2
from datetime import datetime
import pandas as pd
import multiprocessing as mp
import time
import psycopg2
from datetime import datetime
import pandas as pd
import multiprocessing as mp
import time
from multiprocessing import Pool
import re
import psycopg2
import psycopg2
import csv
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import math
import datetime
import pyproj
from pyproj import Proj, transform
import numpy as np
import pandas as pd
#import shapefile
import pyproj
import numpy as np
from pyproj import Transformer, transform
import os
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
from math import acos, cos, asin, sin, atan2, tan, radians
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from functools import reduce
import sys
import multiprocessing, logging
from multiprocessing import Queue
import threading
from queue import Queue
from multiprocessing import Process,Manager

host_ = '130.95.198.59'
port_= 5432
DB_='gswa_dh'
user_ = 'postgres'
pwd_ = 'loopie123pgpw'

# Define an output queue
output = multiprocessing.Queue()
count=0

Comments_fuzzy =[]
Attr_val_fuzzy =[]
cleanup_dic_list =[]
Litho_dico =[]
Attr_val_Dic =[]
Comments_dic =[]
Comments_fuzzy =[]
First_Filter_list=[]
Comments_dic_tmp = []
#process_list_final =[]
Final_fuzzy =[]
   # make global


Final_split_proc_list =[]
Comm_split_proc_list =[]

T_count=0

Tot_workers = mp.cpu_count()
worker_proc = Tot_workers 


def my_range(start, end, step):
    while start < end:
        yield start
        start += step

def Comments_Dic(minlong,maxlong,minlat,maxlat):
    '''
    Function selects the distinct attribute column and attribute value which matches in thesaurus 'thesaurus_geology_comment' with the given region
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - List with extracted data matching attribute column and thesaurus.
    '''
    ''' in query # only attribute value and not attributecol '''
    query = """Select DISTINCT ON (t1.attributevalue)        
    t1.attributecolumn, t1.attributevalue
		 from public.dhgeologyattr t1 
		 inner join public.dhgeology t2 
		 on t1.dhgeologyid = t2.id 
		 inner join collar t3 
		 on t3.id = t2.collarid
		 inner join public.thesaurus_geology_comment t6
		 on t1.attributecolumn = t6.attributecolumn
		 WHERE(t3.longitude BETWEEN %s AND %s) AND (t3.latitude BETWEEN %s AND %s)"""
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query,Bounds)

    
    for record in cur:
        Comments_dic_tmp.append(record)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query, bounds)
   
    #with open('Dic_Comments.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
    
    
    cur.close()
    conn.close()

def Attr_Val_Dic():
    '''
    Funtion extracts Attribute value dictionary table from DB.
    '''
    
    query = '''select * from thesaurus_geology_lithology_code'''
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        Attr_val_Dic.append(record)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
    #with open('Dic_attr_val.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
    

    cur.close()
    conn.close()

def Litho_Dico():
    '''
    Function Extracts Dictionary for lithology from DB.
    '''
    
    query = ''' select thesaurus_geology_hierarchy.fuzzuwuzzy_terms  from thesaurus_geology_hierarchy '''
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
    #print(cur)
    for record in cur:
        #print(record)
        Litho_dico.append(record)
        #print(Litho_dico)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('Dic_litho.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
        
    #print(Litho_dico)
    cur.close()
    conn.close()


def Clean_Up():
    '''
    Function extracts clean up dictionary from DB.
    '''
    
    query = ''' select thesaurus_cleanup.cleanup_term  from thesaurus_cleanup '''
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        cleanup_dic_list.append(record)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('cleanup_dic.csv', 'w',encoding="utf-8") as f:
        #cur.copy_expert(outputquery, f)
        

    cur.close()
    conn.close()


def clean_text(text):
    '''
    Function clean the text by symbols and specified text, uses cleanup dictionary
    Input: 
         - Text
    output: 
        - Cleaned text

    '''
    text=text.lower().replace('unnamed','').replace('meta','').replace('meta-','').replace('undifferentiated ','').replace('unclassified ','').replace(' undifferentiated ','')
    text=text.replace('differentiated','').replace('undiff','').replace('undiferentiated','').replace('undifferntiates','').replace(' undivided','')
    text=(re.sub('\(.*\)', '', text)) # removes text in parentheses
    text=(re.sub('\[.*\]', '', text)) # removes text in parentheses
    text=text.replace('>','').replace('?','').replace('/',' ')
    text=text.lstrip().rstrip()   #strip of left and right spaces
    text = re.sub('\s+', ' ', text)  # for multiple spaces replace by one space
    text = text.replace('>' , ' ')
    text = text.replace('<', ' ')
    text = text.replace('/', ' ')
    text = text.replace(' \' ', ' ')
    text = text.replace(',', ' ')
    text = text.replace('%', ' ')
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    #text = text.replace('', ' ')
    #text = text.replace('+', '')
    text = text.replace('\'', ' ') 
    if text.isnumeric():
        text = re.sub('\d', ' ', text) #replace numbers
    text = text.replace('&' , ' ')
    text = text.replace(',', ' ')
    text = text.replace('.', ' ')
    text = text.replace(':', ' ')
    text = text.replace(';', ' ')
    text = text.replace('$', ' ')
    text = text.replace('@', ' ')
    text = text.replace('\'', '')
    text = text.replace('\\', '')  #replace backslash by space                      
	
    for cleanup_dic_ele in cleanup_dic_list:
        cleaned_item =str(cleanup_dic_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
        text = text.replace('cleaned_item','')
    return text

lemma = nltk.WordNetLemmatizer()

extra_stopwords = [
    'also',
]
stop = stopwords.words('english') + extra_stopwords


def tokenize(text, min_len=1):
    '''Function that tokenize a set of strings
    Input:
        -text: set of strings
        -min_len: tokens length
    Output:
        -list containing set of tokens'''

    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    for token in tokens:
        if token.isalpha() and len(token) >= min_len:
            filtered_tokens.append(token)

    return [x.lower() for x in filtered_tokens if x not in stop]


def tokenize_and_lemma(text, min_len=0):
    '''Function that retrieves lemmatised tokens
    Inputs:
        -text: set of strings
        -min_len: length of text
    Outputs:
        -list containing lemmatised tokens'''
    filtered_tokens = tokenize(text, min_len=min_len)

    lemmas = [lemma.lemmatize(t) for t in filtered_tokens]
    return lemmas




def Attr_val_With_fuzzy():
    '''
    Function gets the fuzzuwuzzy string of the lithology text .The lithology text is cleaned,lemmatised and tokenized.
    Input: Dictionaries Extracted
    Output: is a List and csv file of fuzzywuzzy with score for lithology.
    '''
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    attr_val_sub_list=[]
    #p = re.compile(r'[' ']')
    fieldnames=['CollarID','code','Attr_val','cleaned_text','Fuzzy_wuzzy','Score']
    out= open("Attr_val_fuzzy.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for Attr_val_Dic_ele in Attr_val_Dic_new:
        

       
        cleaned_text_1=clean_text(Attr_val_Dic_ele[2])
        cleaned_text_1=tokenize_and_lemma(cleaned_text_1)
        cleaned_text=" ".join(str(x) for x in cleaned_text_1)  #join each word as string with space

        #cleaned_text=clean_text(Attr_val_Dic_ele[2])  # for logging
        #cleaned_text =  cleaned_text.replace(' rock ',' rocks')   # to handle rock and rocks to get proper fuzzywuzzy
        #cleaned_text =  cleaned_text.replace(' rock',' rocks')  
        if  ' rock ' in cleaned_text :
            cleaned_text =  cleaned_text.replace(' rock ',' rocks ')   # to handle rock and rocks to get proper fuzzywuzzy
        elif ' rock' in cleaned_text:
            cleaned_text =  cleaned_text.replace(' rock',' rocks ') 
        words=(re.sub('\(.*\)', '', cleaned_text)).strip()
        
        #words =  words.replace(' rock',' rocks')   # for mafic rock to get as mafic in csv , since tokenization removes it.
        #if (words == 'mafic rock'):
         #   print(words)
        
        words=words.rstrip('\n\r').split(" ")
        last=len(words)-1 #position of last word in phrase
        for Litho_dico_ele in Litho_dico:
            litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').split(" ")
            
            scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
            for sc in scores:                        
                if(sc[1]>bestmatch): #better than previous best match
                    bestmatch =  sc[1]
                    bestlitho=litho_words[0]
                    top.append([sc[0],sc[1]])
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestmatch=bestmatch*1.01
                        
                elif (sc[1]==bestmatch): #equal to previous best match
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestlitho=litho_words[0]
                        bestmatch=bestmatch*1.01
                        
                    else:
                        
                        top.append([sc[0],sc[1]])
        
        
        #if (words == 'mafic rock'):
            #print(words)     
        if bestmatch >80:
            
            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
            
            #attr_val_sub_list.clear()
            
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
                    #out.write('%d,' %top_new[1])
            out.write('%d,' %bestmatch)
            out.write('\n')
                    #top_new[:] =[]
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
           
            
        else:
            
            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,'Other',bestmatch])  #top_new[1]])
            
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('Other,')
                    #out.write('%d,' %top_new[1])
            out.write('%d,' %bestmatch)
            out.write('\n')
                    #top_new[:] =[]
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''






def Final_Lithology_With_Comments_1():   #DB_lithology_With_Comments_Final_Export,minlong,maxlong,minlat,maxlat):
    '''
    Function Extracts data from tables dhgeologyattr,dhgeology,collar,clbody and attribute column lithology table from DB for the specified region.
    Also joins extraction of Comments attribute column with Comments attribute value .
    For Each row extracted, the from and to depth values are validated , generated fuzzywuzzy values for the lithology along with the score are printed .
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - csv file with the extracted data with fuzzywuzzy and score for lithology and comments.
    '''
    query = ''' SELECT m1.companyid, m1.collarid, m1.fromdepth, m1.todepth, m1.lith_attributecolumn, m1.lith_attributevalue, 
                m2.comments_attributecolumn, m2.comments_attributevalue 
                FROM 
                (select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
                 AS lith_attributecolumn, t1.attributevalue AS lith_attributevalue 
                 from public.dhgeologyattr t1
                 inner join public.dhgeology t2 
                 on t1.dhgeologyid = t2.id 
                 inner join collar t3 
                 on t3.id = t2.collarid 
                 inner join clbody t4 
                 on t4.companyid = t3.companyid
                 inner join public.thesaurus_geology_lithology t5
                 on t1.attributecolumn = t5.attributecolumn
                 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN -30.5 AND -27.5) 
                 ORDER BY t3.companyid ASC) m1
                 FULL JOIN		 
                (select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
                 AS comments_attributecolumn, t1.attributevalue AS comments_attributevalue  
                 from public.dhgeologyattr t1 
                 inner join public.dhgeology t2 
                 on t1.dhgeologyid = t2.id 
                 inner join collar t3 
                 on t3.id = t2.collarid 
                 inner join clbody t4 
                 on t4.companyid = t3.companyid
                 inner join public.thesaurus_geology_comment t6
                 on t1.attributecolumn = t6.attributecolumn
                 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN -30.5 AND -27.5) 
                 ORDER BY t3.companyid ASC) m2 
                 on m1.dhgeologyid = m2.dhgeologyid'''
                 
                 
                 
        
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    print("connected")
    cur = conn.cursor()
    #Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query)  #,Bounds)

    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Company_Lithocode','Company_Lithology','CET_Lithology','Score', 'Comment', 'CET_Comment', 'Comment_Score']
    out= open('DB_lithology_With_Comments_Final_Export.csv', "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    
   # List_File=open('List_Appended.txt','w')
    #print(cur)
    First_Filter_list = [list(elem) for elem in cur]
    #my_df = pd.DataFrame(First_Filter_list)
    #my_df.to_csv('First_Filter.csv', index=False, header=False)
    length_list= len(First_Filter_list)
    for First_filter_ele in First_Filter_list:
        if (First_filter_ele[0] == None and First_filter_ele[1]== None  and First_filter_ele[2]== None  and First_filter_ele[3]== None) or  (First_filter_ele[2] == None and First_filter_ele[3] ==None) :   # for empty fields, bug
            continue
        else :
            First_filter_ele[2],First_filter_ele[3] =Depth_validation_comments(First_filter_ele[2],First_filter_ele[3])  #  ,First_filter_ele[1],First_filter_ele[6],logger1) # validate depth
            CompanyID=First_filter_ele[0]
            CollarID=First_filter_ele[1]
            FromDepth=First_filter_ele[2]
            ToDepth=First_filter_ele[3]
            Company_Lithocode=""
            Company_Lithology=""
            CET_Lithology=""
            Score=0
            Comment=""
            CET_Comment=""
            Comment_Score=0
        
        
        for Attr_val_fuzzy_ele in Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                Company_Lithocode=Attr_val_fuzzy_ele[1]
                Company_Lithology=Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Lithology=Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Score=Attr_val_fuzzy_ele[5]
                
        for Comments_fuzzy_ele in Comments_fuzzy:
            if Comments_fuzzy_ele[1] == First_filter_ele[7]:
                Comment=Comments_fuzzy_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Comment=Comments_fuzzy_ele[3].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Comment_Score=Comments_fuzzy_ele[4]

        out.write('%d,' %CompanyID)
        out.write('%d,' %CollarID)
        out.write('%d,' %FromDepth)
        out.write('%s,' %ToDepth)
        out.write('%s,' %Company_Lithocode)
        out.write('%s,' %Company_Lithology)
        out.write('%s,' %CET_Lithology)
        out.write('%d,' %Score)
        out.write('%s,' %Comment)
        out.write('%s,' %CET_Comment)
        out.write('%d,' %Comment_Score)
        out.write('\n')

    cur.close()
    conn.close()
    out.close()







def Final_Lithology_With_Comments_Split():   #DB_lithology_With_Comments_Final_Export,minlong,maxlong,minlat,maxlat):
    '''
    Function Extracts data from tables dhgeologyattr,dhgeology,collar,clbody and attribute column lithology table from DB for the specified region.
    Also joins extraction of Comments attribute column with Comments attribute value.The extracted data is split using split funtion to create processes.
    
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - csv file with the extracted data with fuzzywuzzy and score for lithology and comments.
    '''
    query = ''' SELECT m1.companyid, m1.collarid, m1.fromdepth, m1.todepth, m1.lith_attributecolumn, m1.lith_attributevalue, 
                m2.comments_attributecolumn, m2.comments_attributevalue 
                FROM 
                (select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
                 AS lith_attributecolumn, t1.attributevalue AS lith_attributevalue 
                 from public.dhgeologyattr t1
                 inner join public.dhgeology t2 
                 on t1.dhgeologyid = t2.id 
                 inner join collar t3 
                 on t3.id = t2.collarid 
                 inner join clbody t4 
                 on t4.companyid = t3.companyid
                 inner join public.thesaurus_geology_lithology t5
                 on t1.attributecolumn = t5.attributecolumn
                 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN -30.5 AND -27.5) 
                 ORDER BY t3.companyid ASC) m1
                 FULL JOIN		 
                (select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
                 AS comments_attributecolumn, t1.attributevalue AS comments_attributevalue  
                 from public.dhgeologyattr t1 
                 inner join public.dhgeology t2 
                 on t1.dhgeologyid = t2.id 
                 inner join collar t3 
                 on t3.id = t2.collarid 
                 inner join clbody t4 
                 on t4.companyid = t3.companyid
                 inner join public.thesaurus_geology_comment t6
                 on t1.attributecolumn = t6.attributecolumn
                 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN -30.5 AND -27.5) 
                 ORDER BY t3.companyid ASC) m2 
                 on m1.dhgeologyid = m2.dhgeologyid'''
                 
                 
                 
        
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    print("connected")
    cur = conn.cursor()
    #Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query)  #,Bounds)
    First_Filter_list = [list(elem) for elem in cur]
    filename_final = 'final_split_list'
    Comments_Litho_Dic_split(First_Filter_list,filename_final,Final_split_proc_list)
    cur.close()
    conn.close()
    




    



def Depth_validation_comments(row_2,row_3) :   #,collarid,dhsurveyid):
    '''
    Funtion validates the from and to depth values according to the requirment
    Input : 
        - From Depth
        - To Depth
    Output:
        - From Depth,To Depth : Right Depth values for from and to depth 
    '''
      
    from_depth = row_2               
    to_depth = row_3
    if (from_depth is not None and to_depth is not None) or  (from_depth is not None or to_depth is not None) :
        if(to_depth == 'NULL' or to_depth == None):
            to_depth = from_depth +0.1
            return from_depth,to_depth
        elif to_depth>from_depth:
            return row_2,row_3
        elif from_depth == to_depth:
            to_depth = to_depth+0.01
            row_3=to_depth
            return row_2,row_3
        elif from_depth >to_depth:
            row_2=to_depth       
            row_3=from_depth
            return row_2,row_3












def Final_comments_with_fuzzy_Process(split_List,Comments_fuzzy,Attr_val_fuzzy,q2,filename):
    '''
        For Each row extracted for a region, the from and to depth values are validated , generated fuzzywuzzy values for the lithology along with the score are printed.
    Inputs:
            -split_List : Each split list to get fuzzywuzzy.
            -Comments_fuzzy : copy of comments fuzzy
            - Attr_val_fuzzy : copy of att_val fuzzy
            - q2 : multiprocessing queue to put the fuzzywuzzy resuts .
            -filename : Print each split output to a csv file.
    '''

    final_fuzzy_list =[]
    for First_filter_ele in split_List:
        if (First_filter_ele[0] == None and First_filter_ele[1]== None  and First_filter_ele[2]== None  and First_filter_ele[3]== None) or  (First_filter_ele[2] == None and First_filter_ele[3] ==None) :   # for empty fields, bug
            continue
        else :
            First_filter_ele[2],First_filter_ele[3] =Depth_validation_comments(First_filter_ele[2],First_filter_ele[3])  #  ,First_filter_ele[1],First_filter_ele[6],logger1) # validate depth
            CompanyID=First_filter_ele[0]
            CollarID=First_filter_ele[1]
            FromDepth=First_filter_ele[2]
            ToDepth=First_filter_ele[3]
            Company_Lithocode=""
            Company_Lithology=""
            CET_Lithology=""
            Score=0
            Comment=""
            CET_Comment=""
            Comment_Score=0
        
        
        for Attr_val_fuzzy_ele in Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                Company_Lithocode=Attr_val_fuzzy_ele[1]
                Company_Lithology=Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Lithology=Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Score=Attr_val_fuzzy_ele[5]
                
        for Comments_fuzzy_ele in Comments_fuzzy:
            if Comments_fuzzy_ele[1] == First_filter_ele[7]:
                Comment=Comments_fuzzy_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Comment=Comments_fuzzy_ele[3].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Comment_Score=Comments_fuzzy_ele[4]

        final_fuzzy_list.append([CompanyID,CollarID,FromDepth,ToDepth,Company_Lithocode,Company_Lithology,CET_Lithology,Score,Comment,CET_Comment,Comment_Score])

    my_df11 = pd.DataFrame(final_fuzzy_list)  #, index=var_name1.keys())
    my_df11.to_csv(filename, index=False, header=True)
    q2.put(final_fuzzy_list)
       



def Comments_Dic_split():
    '''
    Function split the comments dictionary equal to the number of logical process considered 
    Input : 
        Comments dictionary
    Output:
        - Comments Dictionary splits
    '''
    
    Comments_Dic_new = [list(elem) for elem in Comments_dic]
    #print(Comments_Dic_new)
    length_list= len(Comments_Dic_new)
    partition_List = length_list / worker_proc
    actual_part_num = round(partition_List)    # split value with avilable logical process
    print(length_list)
    count=0
    if length_list > 0:      
        x=0
        y=length_list
        for i in range(x,y,actual_part_num):       
            x=i
            count = count + 1
            if count  == worker_proc  :  ## to merge last split with previous one as it is small
                total_split_val =(round(partition_List) * worker_proc)
                diff = length_list - total_split_val

                if diff > 0 or diff == 0 :
                    final_split = x+actual_part_num+diff
                    #print(final_split)
                    globals()['comment_split_list'+ '_' + str(i)] = Comments_Dic_new[x:final_split]
                    variable_list.append(globals()['comment_split_list'+ '_' + str(i)])
                    #print("in final -1")
                    break           # exit after last split , since we added left out records

                elif diff < 0 :
                    final_split = x+ actual_part_num+diff 
                    globals()['comment_split_list'+ '_' + str(i)] = Comments_Dic_new[x:final_split]
                    variable_list.append(globals()['comment_split_list'+ '_' + str(i)])
                    #print("in final-2")
                    break
               
            else:
                globals()['comment_split_list'+ '_' + str(i)] = Comments_Dic_new[x:x+actual_part_num]
                variable_list.append(globals()['comment_split_list'+ '_' + str(i)])
                #print(" Not in final")
            

    #print(count) 

    
    part_num = actual_part_num
    partnum1= part_num
    tot_partnum = part_num
    for x in range(0, count, 1):
        if x > 0 :
                
            var_name1 = 'comment_split_list'+ '_' + str(tot_partnum)
            print(var_name1)
            my_df1 = pd.DataFrame(globals()[var_name1])  
            file_name1 = var_name1 + '.csv'
            my_df1.to_csv(file_name1, index=False, header=True)
            tot_partnum = tot_partnum + part_num
                
        elif x == 0:
            var_name2 = 'comment_split_list' + '_' + str(x)
            print(var_name2)
            my_df2 = pd.DataFrame(globals()[var_name2])   
            file_name2 = var_name2 + '.csv'
            my_df2.to_csv(file_name2, index=False, header=True)
                






def Comments_Litho_Dic_split(dic_litho_comments,filename,Process_list):
    '''
    Function split listoflist to  the number of logical process considered 
    Input : 
         - dic_litho_comments : input  which needs to be split.
         -filename : Each split is printed to a file for verification.
         - Process_list : list to hold the split variables name for later use.
    Output:
        - Comments Dictionary splits in globals variables and in csv file.
    '''
    
    
    length_list= len(dic_litho_comments)
    partition_List = length_list / worker_proc
    actual_part_num = round(partition_List)    # split value with avilable logical process
    print(length_list)
    count=0
    if length_list > 0:      
        x=0
        y=length_list
        for i in range(x,y,actual_part_num):       
            x=i
            count = count + 1
            if count  == worker_proc  :  ## to merge last split with previous one as it is small
                total_split_val =(round(partition_List) * worker_proc)
                diff = length_list - total_split_val

                if diff > 0 or diff == 0 :
                    final_split = x+actual_part_num+diff
                    #print(final_split)
                    globals()[filename+ '_' + str(i)] = dic_litho_comments[x:final_split]
                    Process_list.append(globals()[filename+ '_' + str(i)])
                    #print("in final -1")
                    break           # exit after last split , since we added left out records

                elif diff < 0 :
                    final_split = x+ actual_part_num+diff 
                    globals()[filename+ '_' + str(i)] = dic_litho_comments[x:final_split]
                    Process_list.append(globals()[filename+ '_' + str(i)])
                    #print("in final-2")
                    break
               
            else:
                globals()[filename+ '_' + str(i)] = dic_litho_comments[x:x+actual_part_num]
                Process_list.append(globals()[filename+ '_' + str(i)])
                #print(" Not in final")
            

    #print(count) 

    
    part_num = actual_part_num
    partnum1= part_num
    tot_partnum = part_num
    for x in range(0, count, 1):
        if x > 0 :
                
            var_name1 = filename+ '_' + str(tot_partnum)
            print(var_name1)
            my_df1 = pd.DataFrame(globals()[var_name1])  
            file_name1 = var_name1 + '.csv'
            my_df1.to_csv(file_name1, index=False, header=True)
            tot_partnum = tot_partnum + part_num
                
        elif x == 0:
            var_name2 = filename + '_' + str(x)
            print(var_name2)
            my_df2 = pd.DataFrame(globals()[var_name2])   
            file_name2 = var_name2 + '.csv'
            my_df2.to_csv(file_name2, index=False, header=True)








def Comments_With_fuzzy(q,comment_split, Litho_dico,file_name): 
    '''
    Function find the fuzzywuzzy and score to the comments attribute value 
    Input : 
        q - To fill the fuzzywuzzy results from each process.
        comments_split - comments split to get fuzzywuzzy.
        Litho_Dico - pass Litho_Dico to get fuzzywuzzy
        file_name- print each fuzzywuzzy to a csv file for varification.
    Output:
        - List with fuzzywuzzy and score for comments attribute value.
    '''
    
    
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    Comments_fuzzy_Sub = []
    Comments_Dic_new = [list(elem) for elem in comment_split]
    for Comments_Dic_ele in Comments_Dic_new:
        cleaned_text=clean_text(Comments_Dic_ele[1])
        words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
        words=words.rstrip('\n\r').split(" ")
        last=len(words)-1 #position of last word in phrase
        
        for litho_dico_ele in Litho_dico:
            litho_words=str(litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').split(" ")

            scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
            for sc in scores:                        
                if(sc[1]>bestmatch): #better than previous best match
                    bestmatch =  sc[1]
                    bestlitho=litho_words[0]
                    top.append([sc[0],sc[1]])
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestmatch=bestmatch*1.01
                elif (sc[1]==bestmatch): #equal to previous best match
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestlitho=litho_words[0]
                        bestmatch=bestmatch*1.01
                    else:
                        top.append([sc[0],sc[1]])

        i=0
        if bestmatch >80:
            Comments_fuzzy_Sub.append([Comments_Dic_ele[0],Comments_Dic_ele[1],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
        else:
            Comments_fuzzy_Sub.append([Comments_Dic_ele[0],Comments_Dic_ele[1],cleaned_text,'Other',bestmatch])  #top_new[1]])
            


    #print(Comments_fuzzy_Sub)
    my_df2 = pd.DataFrame(Comments_fuzzy_Sub , columns = ['Comments_Field','Comment_Attr_val','Comment_cleaned_text','Comment_Fuzzy_wuzzy','Comment_Score'])
    my_df2.to_csv(file_name, index=False, header=True)
    q.put(Comments_fuzzy_Sub)
    



def somefun(q,h,w):    #split,litho_Dico):   #,f_name):
    print("child process started")
    #print(len(split))
    #print(split)
    #for litho_dico_ele in litho_Dico:
    #return litho_Dico
    #q.put(h)  # + ' ' + w)
    q.put(w)

def f(x):
    print("started")
    print(x)
    return x

if __name__ == '__main__':
      
    Start_Time = datetime.now()
    print("Stared")
    Attr_Val_Dic()
    Litho_Dico()
    Clean_Up()
    minlong=115.5
    maxlong=118
    minlat=-30.5
    maxlat=-27.5
    Comments_Dic(minlong,maxlong,minlat,maxlat)
    
    
    Attr_val_With_fuzzy()
    
    Start_Time_1 = datetime.now()
    Comments_dic = [list(elem) for elem in Comments_dic_tmp]
   

    filename_comm_split = 'Comment_split_list'    
    Comments_Litho_Dic_split(Comments_dic,filename_comm_split,Comm_split_proc_list)

   

    Process_list = []
    Comments_fuzzy1 = []
    Comments_fuzzy2=[]
    Process_list_final =[]
    Final_fuzzy1 =[]
    Final_fuzzy2=[]
    print(len(Comm_split_proc_list))
    
    #print(len(variable_list[0]))

    
    #print(len(variable_list[1]))
    #print(len(variable_list[2]))
    #print(len(variable_list[3]))
    #print(len(variable_list[4]))
    #print(len(variable_list[5]))
    #print(len(variable_list[6]))
    #print(len(variable_list[7]))

    
    
    #create processes for the number of split in comments to find fuzzywuzzy
    q1 = mp.Queue()
    x=1

    for list_ele in Comm_split_proc_list:
        try:
            outfile = "Fuzzy_"+str(x)+".csv" 
            p = Process(target=Comments_With_fuzzy, args =(q1,list_ele, Litho_dico,outfile)) 
            p.start()
            print("process started")
            Process_list.append(p)
            x = x+ 1
        except:
            raise
            print("Error: unable to start thread", list_ele)


    results = [q1.get() for process in Process_list]
    #print(results)
    Comments_fuzzy1.append(results)
    #print(Comments_fuzzy)


    Comments_fuzzy2 = [val for sublist in Comments_fuzzy1 for val in sublist]    
    #print(flattened1)


    Comments_fuzzy = [val for sublist in Comments_fuzzy2 for val in sublist]    
    #print(flattened2)
    print("got results")

    # wait for processes to finish
    for process in Process_list:
        process.join()
        print("Join")

    

    my_df2 = pd.DataFrame(Comments_fuzzy)  #, index=var_name1.keys())
    final_Fuzzy_Name = 'Final_Fuzzywuzzy_With_Comments.csv'
    my_df2.to_csv(final_Fuzzy_Name, index=False, header=True)

    End_Time_1 = datetime.now()
    print("Process time:", End_Time_1-Start_Time_1)

    #### final funtion 
    Final_Lithology_With_Comments_Split()
    print("final_variable_list  length")
    print(len(Final_split_proc_list))
    #### process creation for final funtion

    Start_Time_2 = datetime.now()
    
    #create processes for the number of split in comments to find fuzzywuzzy
    q2 = mp.Queue()
    x=1

    for final_list_ele in Final_split_proc_list:
        try:
            outfile = "Final_Litho_Comments"+str(x)+".csv" 
            p = Process(target=Final_comments_with_fuzzy_Process, args =(final_list_ele, Comments_fuzzy,Attr_val_fuzzy,q2,outfile)) 
            p.start()
            print("process started")
            Process_list_final.append(p)
            x = x+ 1
        except:
            raise
            print("Error: unable to start thread", list_ele)


    results1 = [q2.get() for process in Process_list_final]
    #print(results)
    Final_fuzzy1.append(results1)
    #print(Comments_fuzzy)


    Final_fuzzy2 = [val for sublist in Final_fuzzy1 for val in sublist]    
    #print(flattened1)


    Final_fuzzy = [val for sublist in Final_fuzzy2 for val in sublist]    
    #print(flattened2)
    print("got results")

    # wait for processes to finish
    for process in Process_list_final:
        process.join()
        print("Join")

    

    my_df_final = pd.DataFrame(Final_fuzzy,columns=['Company_ID','CollarID','Fromdepth','Todepth','Company_Lithocode','Company_Lithology','CET_Lithology','Score','Comment','CET_Comment','Comment_Score'])  #, index=var_name1.keys())
    final_Litho_Comments_Name = 'Final_Litho_With_Comments.csv'
    my_df_final.to_csv(final_Litho_Comments_Name, index=False, header=True)
   

    End_Time_2 = datetime.now()
    print("Final Process time:", End_Time_2-Start_Time_2)


    

   

  

