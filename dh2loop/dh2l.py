import pandas as pd
import geopandas as gpd
import numpy as np
#import psycopg2
import csv
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import math 
from collections import Counter

Attr_col_collar_dic_list=[]

def collar_colar_attri_Final(collar,collarattr):
    fieldnames=['CollarID','HoleId','Longitude','Latitude','RL','MaxDepth']
    out= open("../data/export/DB_Collar_Export.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Pre_id = 0
    Pre_hole_id = ''
    Pre_Longitude =0.0
    Pre_latitude = 0.0
   
    Cur_id = 0
    Cur_hole_id = ''
    Cur_Longitude =0.0
    Cur_latitude = 0.0
   
    list_rl= []
    list_maxdepth =[]
    RL =''
    Maxdepth =''
    write_to_csv = False
    
    cur = collar.set_index ('id').join(collarattr.set_index('collarid'), rsuffix='2')
    del cur['anumber']
    del cur['dataset']
    del cur['companyholeid']
    del cur['companyid']
    del cur['istransformed']
    del cur['modifieddate']
    del cur['modifiedby']
    del cur['mrtfileid']
    del cur['holetype']
    del cur['maxdepth']
    del cur['geom']
    del cur['id']
    del cur['modifieddate2']
    del cur['modifiedby2']
    del cur['mrtdetailid']
    cur ['id']=cur.index
    cols = cur.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cur = cur[cols] 
    cur.reset_index(level=0, inplace=True)
    del cur['index']
    cur=cur.values.tolist()
    
    collar_collarAttr_Filter = [list(elem) for elem in cur]
    DicList_collar_collarattr = [list(elem) for elem in Attr_col_collar_dic_list]
    for colar_ele in collar_collarAttr_Filter:
        for Dic_ele in DicList_collar_collarattr:
            if(colar_ele[4] == Dic_ele[0]): 
                if(Dic_ele[1] == 'rl'):
                    if(Pre_id== colar_ele[0] or Pre_id ==0 or Cur_id ==colar_ele[0]):
                        list_rl.append(Parse_Num(colar_ele[5]))
                        Pre_id =colar_ele[0]
                        Pre_hole_id = colar_ele[1]
                        Pre_Longitude =colar_ele[2]
                        Pre_latitude = colar_ele[3]         
                    else:
                        if(len(list_rl)!=0):
                            RL = maximum(list_rl,'NAN')
                        else:
                            RL = maximum(list_rl,'NAN')
                        if(len(list_maxdepth)!=0):
                            Maxdepth = maximum(list_maxdepth,'NAN')
                        else:
                            Maxdepth = maximum(list_maxdepth,'NAN') 
                        write_to_csv = True
                    
                        Cur_id =colar_ele[0]
                        Cur_hole_id = colar_ele[1]
                        Cur_Longitude =colar_ele[2]
                        Cur_latitude = colar_ele[3]

                        list_rl.clear()
                        list_maxdepth.clear()                 
                        list_rl.append(Parse_Num(colar_ele[5]))
                     
                elif(Dic_ele[1]=='maxdepth'):
                    if(Pre_id== colar_ele[0] or Pre_id == 0 or Cur_id ==colar_ele[0] ):
                        if(colar_ele[5][0] == '-'):
                            list_maxdepth.append(Parse_Num(colar_ele[5])*-1)
                        else:
                            list_maxdepth.append(Parse_Num(colar_ele[5]))

                        Pre_id =colar_ele[0]
                        Pre_hole_id = colar_ele[1]
                        Pre_Longitude =colar_ele[2]
                        Pre_latitude = colar_ele[3]
             
                    else:
                        if(len(list_rl)!=0):
                            RL = maximum(list_rl,'NAN')
                        else:
                            RL = maximum(list_rl,'NAN')
                        if(len(list_maxdepth)!=0):
                            Maxdepth = maximum(list_maxdepth,'NAN')
                        else:
                            Maxdepth = maximum(list_maxdepth,'NAN')
                        
                        write_to_csv = True  

                        Cur_id =colar_ele[0]
                        Cur_hole_id = colar_ele[1]
                        Cur_Longitude =colar_ele[2]
                        Cur_latitude = colar_ele[3]
    
                        list_maxdepth.clear()
                        list_rl.clear()
                     
                        list_maxdepth.append(Parse_Num(colar_ele[5]))
                     
        if(write_to_csv == True):
            out.write('%s,' %Pre_id)
            out.write('%s,' %Pre_hole_id)
            out.write('%s,' %Pre_Longitude)
            out.write('%s,' %Pre_latitude)
            out.write('%s,' %RL)
            out.write('%s,' %Maxdepth)
            out.write('\n')
            write_to_csv =False
            RL =''
            Maxdepth =''
            Pre_id = 0
            Cur_id = 0          

        else:
            continue
    out.close()

def Parse_Num(s1):
    s1=s1.lstrip()
    if re.match("^[-+]?[0-9]+$", s1):
        return(int(s1))
    elif re.match("[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?", s1):
        return(float(s1))
    elif s1.isalpha():
        return(None)

def maximum(iterable, default):
    try:
        return str(max(i for i in iterable if i is not None))
    except ValueError:
        return default

def collar_attr_col_dic():
    cur=pd.read_csv('../data/wamex/rl_maxdepth_dic.csv',encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist()
    for rec in cur:
        Attr_col_collar_dic_list.append(rec)  



Attr_col_survey_dic_list=[]

def Survey_Final(dhsurvey,dhsurveyattr):
    fieldnames=['CollarID','Depth','Azimuth','Dip']
    out= open("../data/export/DB_Survey_Export.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
   
    AZI = 0.0
    DIP = 0.0
    Pre_id =0
    b_AZI =False
    b_DIP =False
    b_DEPTH =False
    back_survey_0 =0
    back_survey_1 = 0.0
    
    cur = dhsurvey.set_index ('id').join(dhsurveyattr.set_index('dhsurveyid'), rsuffix='2')
    del cur['mrtdetailid']
    del cur['units']
    del cur['accuracy']
    del cur['loaddate']
    del cur['loadby']
    del cur['modifieddate']
    del cur['modifiedby']
    del cur['mrtfileid']
    del cur['dip']
    del cur['azimuth']
    del cur['id']
    del cur['loaddate2']
    del cur['loadby2']
    del cur['modifieddate2']
    del cur['modifiedby2']
    cur.reset_index(level=0, inplace=True)
    del cur['index']
    cur=cur.values.tolist()
    
    Survey_First_Filter = [list(elem) for elem in cur]
    Survey_dic_list = [list(elem) for elem in Attr_col_survey_dic_list] 
    for survey_ele in Survey_First_Filter:
        for attr_col_ele in Survey_dic_list:
            if (survey_ele[2] == attr_col_ele[0])  :  #AZI or DIP
                if(Pre_id !=survey_ele[0]  and Pre_id !=0):
                    out.write('%s,' %back_survey_0)
                    out.write('%f,' %back_survey_1)
                    out.write('%f,' %AZI)
                    out.write('%f' %DIP)
                    out.write('\n')
                    AZI=0.0
                    DIP = 0.0
                    Pre_id =0
                    back_survey_0 = 0
                    back_survey_1 = 0.0
                if(survey_ele[0] == 124612):
                    print("124612")
                     
                if ('AZI' in attr_col_ele[1] and (Pre_id ==0 or Pre_id ==survey_ele[0])):
                    Pre_id = survey_ele[0]
                    if survey_ele[3].isalpha():
                        continue
                    elif survey_ele[3].replace('.','',1).lstrip('-').isdigit():
                        if float((survey_ele[3]).replace('\'','').strip())  > 360:
                            continue
                        else:
                            AZI = float((survey_ele[3]).replace('\'','').strip().rstrip('\n\r'))
                            back_survey_0 =survey_ele[0]
                           

                if ('DIP' in attr_col_ele[1] and (Pre_id ==survey_ele[0] or Pre_id ==0)) :   #DIP
                    Pre_id = survey_ele[0]
                    if survey_ele[3].isalpha():
                        continue
                    elif survey_ele[3].replace('.','',1).lstrip('-').isdigit():
                        if float((survey_ele[3]).replace('\'','').replace('<','').strip())  > 90:  # combine al skip cases
                            continue
                        elif float((survey_ele[3]).replace('\'','').replace('<','').strip()) < 0 or float((survey_ele[3]).replace('\'','').replace('<','').strip()) == 0 :
                            DIP = float((survey_ele[3]).replace('\'','').replace('<','').strip())
                            back_survey_0 =survey_ele[0]
                        
                if float(survey_ele[1]) < 0 and (Pre_id ==survey_ele[0] or Pre_id ==0):  # depth # chk all corrections
                    Pre_id = survey_ele[0]
                    survey_ele[1] = abs(survey_ele[1])
                    back_survey_0 =survey_ele[0]
                    back_survey_1 = survey_ele[1]
    out.close()
    
def Attr_col_dic():
    cur=pd.read_csv('../data/wamex/survey_dic.csv',encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist()
    for rec in cur:
         Attr_col_survey_dic_list.append(rec)

First_Filter_list=[]
Attr_col_list=[]
Litho_dico=[]
cleanup_dic_list=[]
Att_col_List_copy_tuple=[]
Attr_val_Dic=[]
Attr_val_fuzzy=[]

def Attr_Val_Dic():
    cur=pd.read_csv('../data/wamex/dic_attr_val_lithology_filter.csv',encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist()  
    for record in cur:
        Attr_val_Dic.append(record)

def Litho_Dico():
    cur=pd.read_csv('../data/wamex/litho_dic_1.csv',encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist() 
    for record in cur:
        Litho_dico.append(record)

def Clean_Up():
    cur=pd.read_csv('../data/wamex/cleanup_lithology.csv',encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist() 
    for record in cur:
        cleanup_dic_list.append(record)

def clean_text(text):
    text=text.lower().replace('unnamed','').replace('metamorphosed','').replace('meta','').replace('meta-','').replace('undifferentiated ','').replace('unclassified ','')
    text=text.replace('differentiated','').replace('undiff','').replace('undiferentiated','').replace('undifferntiates','')
    text=(re.sub('\(.*\)', '', text)) # removes text in parentheses
    text=(re.sub('\[.*\]', '', text)) # removes text in parentheses
    text=text.replace('>','').replace('?','').replace('/',' ') 
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

    for cleanup_dic_ele in cleanup_dic_list:
        cleaned_item =str(cleanup_dic_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
        text = text.replace('cleaned_item','')
    return text

def Attr_val_With_fuzzy():
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    attr_val_sub_list=[]
    fieldnames=['CollarID','Company_LithoCode','Company_Litho','Company_Litho_Cleaned','CET_Litho','Score']
    out= open("../data/export/CET_Litho.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for Attr_val_Dic_ele in Attr_val_Dic_new:   
        cleaned_text=clean_text(Attr_val_Dic_ele[2])
        words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
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
                        top.append([sc[0],sc[1]])        
        i=0
        if bestmatch >80:
            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%d,' %bestmatch)
            out.write('\n')
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

def Final_Lithology():
    dic_att_col_lithology=pd.read_csv('../data/wamex/dic_att_col_lithology.csv',encoding = "ISO-8859-1", dtype='object')
    keys=list(dic_att_col_lithology.columns.values)
    dhgeology= pd.read_csv('../data/wamex/dhgeology.csv',encoding = "ISO-8859-1", dtype='object')
    dhgeologyattr= pd.read_csv('../data/wamex/dhgeologyattr.csv',encoding = "ISO-8859-1", dtype='object')
    i1=dhgeologyattr.set_index(keys).index
    i2=dic_att_col_lithology.set_index(keys).index
    dhgeologyattr=dhgeologyattr[i1.isin(i2)]
    cur = dhgeology.set_index ('id').join(dhgeologyattr.set_index('dhgeologyid'), rsuffix='2')
    del cur['units']
    del cur['accuracy']
    del cur['modifieddate']
    del cur['modifiedby']
    del cur['mrtfileid']
    del cur['id']
    del cur['modifieddate2']
    del cur['modifiedby2']
    del cur['mrtdetailid']
    cur = cur.set_index ('collarid').join(collar.set_index('id'), rsuffix='2')
    cur ['collarid']=cur.index
    del cur['holeid']
    del cur['anumber']
    del cur['dataset']
    del cur['companyholeid']
    del cur['longitude']
    del cur['latitude']
    del cur['istransformed']
    del cur['modifieddate']
    del cur['modifiedby']
    del cur['mrtfileid']
    del cur['holetype']
    del cur['maxdepth']
    del cur['geom']
    cols = cur.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cur = cur[cols] 
    cur.reset_index(level=0, inplace=True)
    del cur['index']
    cols = cur.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    cur = cur[cols]
    cur=cur.values.tolist()
    #cur[cur.attributevalue.notnull()]
    
    First_Filter_list = [list(elem) for elem in cur]
    fieldnames=['Company_ID','CollarID','FromDepth','ToDepth','Company_LithoCode','Company_Litho','CET_Litho','Score']
    out= open("../data/export/DB_Lithology_Export.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    for First_filter_ele in First_Filter_list:
        for Attr_val_fuzzy_ele in Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                out.write('%d,' %First_filter_ele[0])
                out.write('%d,' %First_filter_ele[1])
                out.write('%d,' %First_filter_ele[2])
                out.write('%s,' %First_filter_ele[3])
                out.write('%s,' %Attr_val_fuzzy_ele[1])
                out.write('%s,' %Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',',''))
                out.write('%s,' %Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',',''))   #.replace(',' , ''))
                out.write('%d,' %int(Attr_val_fuzzy_ele[5]))
                out.write('\n')
    out.close()
