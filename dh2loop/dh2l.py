import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import numpy as np
import fileinput
#import psycopg2
import csv
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import math 
from math import acos, cos, asin, sin, atan2, tan, radians
from collections import Counter
import pyproj
from pyproj import Proj, transform
import os
import collections
import bisect
import vtk
import vtk.util.numpy_support as vtknumpy
from vtk.numpy_interface import dataset_adapter as vtkdsa

Attr_col_collar_dic_list=[]
def collar_collarattr_final(collar_file, collarattr_file, rl_maxdepth_dic_file, DB_Collar_Export):
    #Attr_col_collar_dic_list=[]
    fieldnames=['CollarID','HoleId','Longitude','Latitude','RL','MaxDepth']
    out= open(DB_Collar_Export, "w",encoding ="utf-8")
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
    collar= pd.read_csv(collar_file,encoding = "ISO-8859-1", dtype='object')
    collarattr= pd.read_csv(collarattr_file,encoding = "ISO-8859-1", dtype='object')
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
    for collar_ele in collar_collarAttr_Filter:
        for Dic_ele in DicList_collar_collarattr:
            if(collar_ele[4] == Dic_ele[0]):
                if(Dic_ele[1] == 'rl'):
                    if(Pre_id== collar_ele[0] or Pre_id ==0 or Cur_id ==collar_ele[0]):
                        list_rl.append(Parse_Num(collar_ele[5]))
                        Pre_id =collar_ele[0]
                        Pre_hole_id = collar_ele[1]
                        Pre_Longitude =collar_ele[2]
                        Pre_latitude = collar_ele[3]         
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
                    
                        Cur_id =collar_ele[0]
                        Cur_hole_id = collar_ele[1]
                        Cur_Longitude =collar_ele[2]
                        Cur_latitude = collar_ele[3]

                        list_rl.clear()
                        list_maxdepth.clear()                 
                        list_rl.append(Parse_Num(collar_ele[5]))
                     
                elif(Dic_ele[1]=='maxdepth'):
                    if(Pre_id== collar_ele[0] or Pre_id == 0 or Cur_id ==collar_ele[0] ):
                        if(collar_ele[5][0] == '-'):
                            list_maxdepth.append(Parse_Num(collar_ele[5])*-1)
                        else:
                            list_maxdepth.append(Parse_Num(collar_ele[5]))

                        Pre_id =collar_ele[0]
                        Pre_hole_id = collar_ele[1]
                        Pre_Longitude =collar_ele[2]
                        Pre_latitude = collar_ele[3]             
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
                        Cur_id =collar_ele[0]
                        Cur_hole_id = collar_ele[1]
                        Cur_Longitude =collar_ele[2]
                        Cur_latitude = collar_ele[3]
    
                        list_maxdepth.clear()
                        list_rl.clear()
                     
                        list_maxdepth.append(Parse_Num(collar_ele[5]))
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

def collar_attr_col_dic(rl_maxdepth_dic_file):
    cur=pd.read_csv(rl_maxdepth_dic_file,encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist()
    for rec in cur:
        Attr_col_collar_dic_list.append(rec) 
        
def convert_coords(row):
    x2,y2 = pyproj.transform(pyproj.Proj({'init': 'EPSG:4326'}),pyproj.Proj({'init': 'EPSG:28350'}),row['Longitude'],row['Latitude'])
    return pd.Series({'X':x2,'Y':y2})
    #https://stackoverflow.com/questions/39620105/converting-between-projections-using-pyproj-in-pandas-dataframe
    
Attr_col_survey_dic_list=[]
def survey_final(dhsurvey_file,dhsurveyattr_file, DB_Survey_Export):
    fieldnames=['CollarID','Depth','Azimuth','Dip']
    out= open(DB_Survey_Export, "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    dhsurvey= pd.read_csv(dhsurvey_file,encoding = "ISO-8859-1", dtype='object')
    dhsurveyattr= pd.read_csv(dhsurveyattr_file,encoding = "ISO-8859-1", dtype='object')
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
    cur.sort_values(['collarid', 'depth'], ascending=[True, True])
    cur=cur.values.tolist()
    #print(cur)
    
    AZI = 0.0
    AZI_list =0.0
    AZI_sub_list=[]
    AZI_DIP_LIST =[]
    AZI_ele = 0.0
    DIP = -90
    Pre_id =0
    b_AZI =False
    b_DIP =False
    b_DEPTH =False
    back_survey_0 =0
    back_survey_1 = -1.1
    One_DIP=False
    One_AZI =False
    
    Survey_First_Filter = [list(elem) for elem in cur]
    Survey_dic_list = [list(elem) for elem in Attr_col_survey_dic_list] 
    for survey_ele in Survey_First_Filter:
        for attr_col_ele in Survey_dic_list:
            if (survey_ele[2] == attr_col_ele[0])  :  #AZI or DIP
                if(Pre_id !=survey_ele[0]  and Pre_id !=0):
                    if(len(AZI_DIP_LIST)!=0):
                        AZI_DIP_Print=[]
                        list_AZI =[]
                        list_DIP =[]
                        if AZI_sub_list:
                            AZI_ele=max(AZI_sub_list)
                        if float(survey_ele[1]) < 0 :
                            survey_ele[1] = abs(survey_ele[1])
                        AZI_DIP_LIST.append([back_survey_1,AZI_ele,DIP])
                        AZI_1 =0.0
                        AZI_2 =0.0
                        DIP_1 =-90
                        DIP_2 =-90
                        for loop1_ele in AZI_DIP_LIST:
                            for loop2_ele in AZI_DIP_LIST:
                                if(loop1_ele[0] == loop2_ele[0]):
                                    if abs(loop1_ele[1]) == abs(loop2_ele[1]) and abs(loop1_ele[2]) == abs(loop2_ele[2]):
                                        AZI_1=loop1_ele[1]
                                        DIP_1 = loop1_ele[2]
                                    elif abs(loop1_ele[1]) != abs(loop2_ele[1]) and abs(loop1_ele[2]) != abs(loop2_ele[2]):
                                        if abs(loop1_ele[1]) < abs(loop2_ele[1]):
                                            AZI_2 = loop1_ele[1]
                                        else:
                                            AZI_2 = loop2_ele[1]
                                        if abs(loop1_ele[2]) < abs(loop2_ele[2]):
                                            DIP_2 = loop1_ele[2]
                                        else:
                                            DIP_2 = loop2_ele[2]
                            if abs(AZI_1) > abs(AZI_2):
                                AZI_ = AZI_1
                            else:
                                AZI_ = AZI_2
                            if abs(DIP_1) < abs(DIP_2):
                                DIP_ = DIP_1
                            else:
                                DIP_ = DIP_2
                            AZI_DIP_Print.append([loop1_ele[0],AZI_,DIP_])
                            AZI_1 =0.0
                            AZI_2 =0.0
                            DIP_1 =-90
                            DIP_2 =-90
                            AZI_= 0.0
                            DIP_ = -90
                        b_set = set(tuple(x) for x in AZI_DIP_Print)
                        AZI_DIP_Print_Filter = [ list(x) for x in b_set]
                        AZI_DIP_Print_Filter = dict((x[0], x) for x in AZI_DIP_Print_Filter).values()
                        One_AZI= False
                        if(len(AZI_DIP_Print_Filter)!=0):
                            for AZI_DIP_Print_Filter_ele in AZI_DIP_Print_Filter:
                                out.write('%s,' %back_survey_0)
                                out.write('%s,' %AZI_DIP_Print_Filter_ele[0])
                                out.write('%f,' %AZI_DIP_Print_Filter_ele[1])
                                out.write('%f,' %AZI_DIP_Print_Filter_ele[2])
                                out.write('\n')
                        AZI_DIP_Print.clear()
                    AZI_DIP_LIST.clear()
                    if(One_AZI==True):
                        out.write('%s,' %back_survey_0)
                        out.write('%s,' %back_survey_1)
                        out.write('%f,' %AZI)
                        out.write('%f,' %DIP)
                        out.write('\n')
                    AZI =0.0
                    DIP =-90
                    #One_DIP =False
                    One_AZI =False
                    AZI_sub_list.clear()
                    AZI_ele =0.0
                    back_survey_0 = 0
                    back_survey_1 = -1.1
                    Pre_id =0
                if ('AZI' in attr_col_ele[1] and (Pre_id ==0 or Pre_id ==survey_ele[0])): # and back_survey_1 == survey_ele[1] ):    #AZI
                    Pre_id = survey_ele[0]
                    if survey_ele[3].isalpha():
                        continue
                    elif survey_ele[3].replace('.','',1).lstrip('-').isdigit():
                        if float((survey_ele[3]).replace('\'','').replace('>','').replace('<','').strip())  > 360:
                            continue
                        else:
                            if (back_survey_1 == survey_ele[1] or back_survey_1==-1.1 ):
                                AZI = float((survey_ele[3]).replace('\'','').strip().replace('<','').replace('>','').rstrip('\n\r'))
                                AZI_sub_list.append(AZI)
                                back_survey_0 =survey_ele[0]
                                back_survey_1 = survey_ele[1]
                                One_AZI =True
                            else:
                                if AZI_sub_list:
                                    AZI_ele=max(AZI_sub_list)
                                if float(survey_ele[1]) < 0:
                                    survey_ele[1] = abs(survey_ele[1])
                                AZI_DIP_LIST.append([back_survey_1,AZI_ele,DIP])
                                AZI_sub_list.clear()
                                AZI_ele =0.0
                                AZI=0.0
                                DIP=-90
                                AZI = float((survey_ele[3]).replace('\'','').strip().rstrip('\n\r'))
                                AZI_sub_list.append(AZI)
                                back_survey_0 =survey_ele[0]
                                back_survey_1 = survey_ele[1]
                                One_AZI =False
                if ('DIP' in attr_col_ele[1] and (Pre_id ==survey_ele[0] or Pre_id ==0)) :    #DIP
                    Pre_id = survey_ele[0]
                    if survey_ele[3].isalpha():
                        continue
                    elif survey_ele[3].replace('.','',1).lstrip('-').isdigit():
                        if float((survey_ele[3]).replace('\'','').replace('<','').strip())  > 90:  # combine al skip cases
                            continue
                        elif float((survey_ele[3]).replace('\'','').replace('<','').strip()) < 0 or float((survey_ele[3]).replace('\'','').replace('<','').strip()) == 0 :
                            if (back_survey_1 == survey_ele[1] or  back_survey_1==-1.1):
                                DIP= float((survey_ele[3]).replace('\'','').replace('<','').replace('>','').strip())
                                back_survey_0 =survey_ele[0]
                                back_survey_1 = survey_ele[1]
                        else:
                            if AZI_sub_list:
                                AZI_ele=max(AZI_sub_list)
                            if float(survey_ele[1]) < 0 :
                                survey_ele[1] = abs(survey_ele[1])
                            AZI_DIP_LIST.append([back_survey_1,AZI_ele,DIP])
                            AZI_sub_list.clear()
                            AZI_ele =0.0
                            DIP=-90
                            AZI=0.0
                            DIP= float((survey_ele[3]).replace('\'','').replace('<','').replace('>','').strip())
                            back_survey_0 =survey_ele[0]
                            back_survey_1 = survey_ele[1]
    out.close()
    dbsurvey= pd.read_csv(DB_Survey_Export)
    dbsurvey= dbsurvey.sort_values(['CollarID', 'Depth'], ascending=[True, True])
    dbsurvey= dbsurvey.loc[:, ~dbsurvey.columns.str.contains('^Unnamed')]
    dbsurvey.to_csv(DB_Survey_Export,index=False)
        
def count_Digit(n):
    if n > 0:
        digits = int(math.log10(n))+1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n))+1 # +1 if you don't count the '-'
  
    return digits

def survey_attr_col_dic(survey_dic_file):
    cur=pd.read_csv(survey_dic_file,encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist()
    for rec in cur:
        Attr_col_survey_dic_list.append(rec)
#survey_attr_col_dic()
#survey_final()
        
def dia2xyz(X1,Y1,Z1,I1,Az1,Distance1,I2,Az2,Distance2):
    I1=radians(I1)
    Az1=radians(Az1)
    I2=radians(I2)
    Az2=radians(Az2)
    
    MD = Distance2 - Distance1

    Beta = acos(cos(I2 - I1) - (sin(I1)*sin(I2)*(1-cos(Az2-Az1))))
    if(Beta==0):
        RF=1
    else:
        RF = 2 / Beta * tan(Beta / 2)

    dX = MD/2 * (sin(I1)*sin(Az1) + sin(I2)*sin(Az2))*RF
    dY = MD/2 * (sin(I1)*cos(Az1) + sin(I2)*cos(Az2))*RF
    dZ = MD/2 * (cos(I1) + cos(I2))*RF

    X2 = X1 + dX
    Y2 = Y1 + dY
    Z2 = Z1 - dZ
    
    return(X2,Y2,Z2)
    
def convert_survey(DB_Collar_Export,DB_Survey_Export, DB_Survey_Export_Calc):
    location=pd.read_csv(DB_Collar_Export)
    #location.astype({'CollarID': '|S80'}).dtypes
    #location.astype({'X': 'int32'}).dtypes
    #location.astype({'Y': 'int32'}).dtypes
    #location.astype({'RL': 'int32'}).dtypes
    location.set_index('CollarID',inplace=True)
    #print(location)
    
    survey=pd.read_csv(DB_Survey_Export)
    survey.astype({'CollarID': '|S80'}).dtypes
    survey.drop(survey.index[len(survey)-1],inplace=True)
    survey.reset_index(inplace=True)
    #print(survey)
    survey=pd.merge(survey,location, how='left', on='CollarID')

    fieldnames=['CollarID','Depth','Azimuth','Dip','Index','X','Y','Z']
    out= open(DB_Survey_Export_Calc, "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    
    holeid=''
    for indx,interval in survey.iterrows():
        if(interval['CollarID'] != holeid):
            first=True
            
        if(first):
            first=False
            lasti=float(interval['Dip'])
            lasta=float(interval['Azimuth'])
            lastd=float(interval['Depth'])
            holeid=(interval['CollarID'])
            X1=float(interval['X'])
            Y1=float(interval['Y'])
            Z1=float(interval['RL'])
            index=indx+2
            
            out.write('%s,' %holeid)
            out.write('%f,' %lastd)
            out.write('%f,' %lasta)
            out.write('%f,' %lasti)
            out.write('%d,' %index)
            out.write('%f,' %X1)
            out.write('%f,' %Y1)
            out.write('%f,' %Z1)
            out.write('\n')
            #print(holeid,float(interval['Depth']),float(interval['Azimuth'],float(interval['Dip']),indx+2,X1,Y1,Z1)
        else:
            if(indx<len(survey) and interval['CollarID'] == holeid):
                X2,Y2,Z2=dia2xyz(X1,Y1,Z1,lasti,lasta,lastd,float(interval['Dip']),float(interval['Azimuth']),float(interval['Depth']))
                out.write('%s,' %holeid)
                out.write('%f,' %lastd)
                out.write('%f,' %lasta)
                out.write('%f,' %lasti)
                out.write('%d,' %index)
                out.write('%f,' %X2)
                out.write('%f,' %Y2)
                out.write('%f,' %Z2)
                out.write('\n')
                #print(holeid,float(interval['Depth']),float(interval['Azimuth'],float(interval['Dip']),interval['index'],X2,Y2,Z2)
                X1=X2
                Y1=Y2
                Z1=Z2
                lasti=float(interval['Dip'])
                lasta=float(interval['Azimuth'])
                lastd=float(interval['Depth'])        
    out.close()
    #http://www.drillingformulas.com/minimum-curvature-method/
    #https://gis.stackexchange.com/questions/13484/how-to-convert-distance-azimuth-dip-to-xyz
    
First_Filter_list=[]
Attr_col_list=[]
Litho_dico=[]
cleanup_dic_list=[]
Att_col_List_copy_tuple=[]
Attr_val_Dic=[]
Attr_val_fuzzy=[]

def litho_attr_col_dic(dic_attr_col_lithology_file,):
    cur=pd.read_csv(dic_attr_col_lithology_file, encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist()
    for record in cur:
        Attr_col_list.append(record)
        
def litho_attr_val_dic(dic_attr_val_lithology_file):
    cur=pd.read_csv(dic_attr_val_lithology_file,encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist() 
    for record in cur:
        Attr_val_Dic.append(record)
    #return(Attr_val_Dic)

def litho_dico(litho_dic_file):
    cur=pd.read_csv(litho_dic_file,encoding = "ISO-8859-1", dtype='object')
    #Litho_dico=cur.values.tolist() 
    cur=cur.values.tolist() 
    for record in cur:
        #cur=cur.values.tolist()
        #record=record.split(' ') 
        Litho_dico.append(record)
    #return(Litho_dico)

def clean_up(cleanup_lithology_file):
    cur=pd.read_csv(cleanup_lithology_file,encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist() 
    for record in cur:
        cleanup_dic_list.append(record)
        
def First_Filter(collar_file, dhgeology_file, dhgeologyattr_file, dic_attr_col_lithology_file):
    dic_attr_col_lithology=pd.read_csv(dic_attr_col_lithology_file,encoding = "ISO-8859-1", dtype='object')
    keys=list(dic_attr_col_lithology.columns.values)
    dhgeology= pd.read_csv(dhgeology_file,encoding = "ISO-8859-1", dtype='object')
    dhgeologyattr= pd.read_csv(dhgeologyattr_file,encoding = "ISO-8859-1", dtype='object')
    i1=dhgeologyattr.set_index(keys).index
    i2=dic_attr_col_lithology.set_index(keys).index
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
    collar= pd.read_csv(collar_file,encoding = "ISO-8859-1", dtype='object')
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
    #print(cur)
    cur=cur.values.tolist() 
    a_list = [list(elem) for elem in cur]
    for row in a_list:
        att_val=row[4]
        for att_col_ele in Attr_col_list:
            dic_att_col=str(att_col_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
            if att_val == dic_att_col :
                from_depth = row[2]
                to_depth = row[3]
                if from_depth is not None and to_depth is not None:
                    if to_depth>from_depth:
                        First_Filter_list.append(row)
                    elif from_depth == to_depth:
                        to_depth = to_depth+0.01
                        row[3]=to_depth
                        First_Filter_list.append(row)
                    elif from_depth >to_depth:
                        row[2]=to_depth
                        row[3]=from_depth
                        First_Filter_list.append(row)

def clean_text(text):
    text=text.lower()
    #.replace('unnamed','').replace('metamorphosed','').replace('metamorphism','').replace('meta','').replace('meta-','')
    #text=text.replace('undifferentiated ','').replace('undiferentiated','').replace('undifferntiates','').replace('differentiated','').replace('undiff','').replace('unclassified ','')
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
        cleaned_item =str(cleanup_dic_ele).replace('(','').replace(')','').replace(',','').replace('\'','').replace('[','').replace(']','')
        text = text.replace(cleaned_item,'').replace('  ',' ')
        if text==" ":
             return cleaned_item
        else:
            return text

def litho_attr_val_with_fuzzy(CET_Litho):
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    attr_val_sub_list=[]
    fieldnames=['CompanyID','Company_LithoCode','Company_Litho','Company_Litho_Cleaned','CET_Litho','Score']
    out= open(CET_Litho, "w",encoding ="utf-8")
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
            litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').replace('[','').replace(']','').split(" ")
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
            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%d,' %bestmatch)
            out.write('\n')
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
            
        else:
            Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,'unclassified_rock',bestmatch])  #top_new[1]])
            out.write('%d,' %int(Attr_val_Dic_ele[0]))
            out.write('%s,' %Attr_val_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))
            out.write('%s,' %Attr_val_Dic_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))     #.replace(',' , '').replace('\n' , ''))
            out.write('%s,' %cleaned_text.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('unclassified_rock,')
            #out.write('%d,' %top_new[1])
            out.write('%d,' %bestmatch)
            out.write('\n')
            #top_new[:] =[]
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
    #print(Attr_val_fuzzy)

Attr_val_fuzzy_list2=[]
def Attr_val_fuzzy2(CET_Litho):
    cur=pd.read_csv(CET_Litho,encoding = "ISO-8859-1", dtype='object')
    cur=cur.values.tolist() 
    for record in cur:
        Attr_val_fuzzy_list2.append(record)
            
    
def litho_final(collar_file, dhgeology_file, dhgeologyattr_file, dic_attr_col_lithology_file, CET_Litho, DB_Lithology_Export):
    dic_attr_col_lithology=pd.read_csv(dic_attr_col_lithology_file,encoding = "ISO-8859-1", dtype='object')
    keys=list(dic_attr_col_lithology.columns.values)
    dhgeology= pd.read_csv(dhgeology_file,encoding = "ISO-8859-1", dtype='object')
    dhgeologyattr= pd.read_csv(dhgeologyattr_file,encoding = "ISO-8859-1", dtype='object')
    i1=dhgeologyattr.set_index(keys).index
    i2=dic_attr_col_lithology.set_index(keys).index
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
    collar= pd.read_csv(collar_file,encoding = "ISO-8859-1", dtype='object')
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
    #print(cur)
    cur=cur.values.tolist() 
    #cur[cur.attributevalue.notnull()]
    for record in cur:
        First_Filter_list.append(record)
    #print(First_Filter_list)
    
    Attr_val_fuzzy2(CET_Litho)
    
    #First_Filter_list = [list(elem) for elem in cur]
    fieldnames=['Company_ID','CollarID','FromDepth','ToDepth','Company_LithoCode','Company_Litho','CET_Litho','Score']
    out= open(DB_Lithology_Export, "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    for First_filter_ele in First_Filter_list:
        #print(First_filter_ele)
        for Attr_val_fuzzy_ele in Attr_val_fuzzy_list2:
            #print(Attr_val_fuzzy)
            if Attr_val_fuzzy_ele[0] == First_filter_ele[0] and Attr_val_fuzzy_ele[1] == First_filter_ele[5]:
                out.write('%s,' %First_filter_ele[0])
                out.write('%s,' %First_filter_ele[1])
                out.write('%s,' %First_filter_ele[2])
                out.write('%s,' %First_filter_ele[3])
                out.write('%s,' %Attr_val_fuzzy_ele[1])
                out.write('%s,' %Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',',''))
                out.write('%s,' %Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',',''))    #.replace(',' , ''))
                out.write('%d,' %int(Attr_val_fuzzy_ele[5]))
                out.write('\n')
    out.close()
def dsmincurb (len12,azm1,dip1,azm2,dip2):
    DEG2RAD = 3.141592654/180.0
    i1 = (90 - float(dip1)) * DEG2RAD
    a1 = float(azm1) * DEG2RAD
    
    i2 = (90 - float(dip2)) * DEG2RAD
    a2 = float(azm2) * DEG2RAD
    
    #Beta = acos(cos(I2 - I1) - (sin(I1)*sin(I2)*(1-cos(Az2-Az1))))
    dl = acos(cos(float(i2)-float(i1))-(sin(float(i1))*sin(float(i2))*(1-cos(float(a2)-float(a1)))))
    if dl!=0.:
        rf = 2*tan(dl/2)/dl  # minimum curvature
    else:
        rf=1                 # balanced tangential

    dz = 0.5*len12*(cos(float(i1))+cos(float(i2)))*rf
    dn = 0.5*len12*(sin(float(i1))*cos(float(a1))+sin(float(i2))*cos(float(a2)))*rf
    de = 0.5*len12*(sin(float(i1))*sin(float(a1))+sin(float(i2))*sin(float(a2)))*rf
    return dz,dn,de
    #modified from pygslib
	
def dia2xyz(X1,Y1,Z1,I1,Az1,Distance1,I2,Az2,Distance2):
    I1=radians(I1)
    Az1=radians(Az1)
    I2=radians(I2)
    Az2=radians(Az2)
    
    MD = Distance2 - Distance1

    Beta = acos(cos(I2 - I1) - (sin(I1)*sin(I2)*(1-cos(Az2-Az1))))
    if(Beta==0):
        RF=1
    else:
        RF = 2 / Beta * tan(Beta / 2)

    dX = MD/2 * (sin(I1)*sin(Az1) + sin(I2)*sin(Az2))*RF
    dY = MD/2 * (sin(I1)*cos(Az1) + sin(I2)*cos(Az2))*RF
    dZ = MD/2 * (cos(I1) + cos(I2))*RF

    X2 = X1 + dX
    Y2 = Y1 + dY
    Z2 = Z1 - dZ
    
    return(X2,Y2,Z2)
	
def interp_ang1D(azm1,dip1,azm2,dip2,len12,d1):
    # convert angles to coordinates
    x1,y1,z1 = ang2cart(azm1,dip1)
    x2,y2,z2 = ang2cart(azm2,dip2)

    # interpolate x,y,z
    x = x2*d1/len12 + x1*(len12-d1)/len12
    y = y2*d1/len12 + y1*(len12-d1)/len12
    z = z2*d1/len12 + z1*(len12-d1)/len12

    # get back the results as angles
    azm,dip = cart2ang(x,y,z)
    return azm, dip
    #modified from pygslib
    
def ang2cart(azm, dip):
    DEG2RAD=3.141592654/180.0
    # convert degree to rad and correct sign of dip
    razm = float(azm) * float(DEG2RAD)
    rdip = -(float(dip)) * float(DEG2RAD)

    # do the conversion
    x = sin(razm) * cos(rdip)
    y = cos(razm) * cos(rdip)
    z = sin(rdip)
    return x,y,z
    #modified from pygslib
    
def cart2ang(x,y,z):
    if x>1.: x=1.
    if x<-1.: x=-1.
    if y>1.: y=1.
    if y<-1.: y=-1.
    if z>1.: z=1.
    if z<-1.: z=-1.
    RAD2DEG=180.0/3.141592654
    pi = 3.141592654
    azm= float(atan2(x,y))
    if azm<0.:
        azm= azm + pi*2
    azm = float(azm) * float(RAD2DEG)
    dip = -(float(asin(z))) * float(RAD2DEG)
    return azm, dip
    #modified from pygslib
    
def angleson1dh(indbs,indes,ats,azs,dips,lpt):
    for i in range (indbs,indes):
        a=ats[i]
        b=ats[i+1]
        azm1 = azs[i]
        dip1 = dips[i]
        azm2 = azs[i+1]
        dip2 = dips[i+1]
        len12 = ats[i+1]-ats[i]
        if lpt>=a and lpt<b:
            d1= lpt- a
            azt,dipt = interp_ang1D(azm1,dip1,azm2,dip2,len12,d1)
            return azt, dipt
    a=ats[indes]
    azt = azs[indes]
    dipt = dips[indes]
    if float(lpt)>=float(a):
        return   azt, dipt
    else:
        return   np.nan, np.nan
    #modified from pygslib
    
def convert_lithology(DB_Collar_Export, DB_Survey_Export, DB_Lithology_Export ,DB_Lithology_Export_Calc):
    collar= pd.read_csv(DB_Collar_Export,encoding = "ISO-8859-1", dtype='object')
    survey= pd.read_csv(DB_Survey_Export,encoding = "ISO-8859-1", dtype='object')
    litho= pd.read_csv(DB_Lithology_Export,encoding = "ISO-8859-1", dtype='object')
    
    collar.CollarID = collar.CollarID.astype(float)
    survey.CollarID = survey.CollarID.astype(float)
    survey.Depth = survey.Depth.astype(float)
    litho.CollarID = litho.CollarID.astype(float)
    litho.FromDepth = litho.FromDepth.astype(float)
    
    collar.sort_values(['CollarID'], inplace=True)
    survey.sort_values(['CollarID', 'Depth'], inplace=True)
    litho.sort_values(['CollarID', 'FromDepth'], inplace=True)
    
    idc =collar['CollarID'].values
    xc = collar['X'].values
    yc = collar['Y'].values
    zc = collar['RL'].values
    ids = survey['CollarID'].values
    ats = survey['Depth'].values
    azs = survey['Azimuth'].values
    dips = survey['Dip'].values
    idt =litho['CollarID'].values
    fromt = litho['FromDepth'].values
    tot = litho['ToDepth'].values
    compid=litho['Company_ID'].values
    complc=litho['Company_LithoCode'].values
    compl=litho['Company_Litho'].values
    cetlit=litho['CET_Litho'].values
    score=litho['Score'].values
    lvl1=litho['Level_1'].values
    lvl2=litho['Level_2'].values
    lvl3=litho['Level_3'].values
    
    nc= idc.shape[0]
    ns= ids.shape[0]
    nt= idt.shape[0]
    
    azmt = np.empty([nt], dtype=float)
    dipmt = np.empty([nt], dtype=float)
    xmt = np.empty([nt], dtype=float)
    ymt = np.empty([nt], dtype=float)
    zmt = np.empty([nt], dtype=float)
    azbt = np.empty([nt], dtype=float)
    dipbt = np.empty([nt], dtype=float)
    xbt = np.empty([nt], dtype=float)
    ybt = np.empty([nt], dtype=float)
    zbt = np.empty([nt], dtype=float)
    azet = np.empty([nt], dtype=float)
    dipet = np.empty([nt], dtype=float)
    xet = np.empty([nt], dtype=float)
    yet = np.empty([nt], dtype=float)
    zet = np.empty([nt], dtype=float)

    azmt[:] = np.nan
    dipmt[:] = np.nan
    azbt [:]= np.nan
    dipbt [:]= np.nan
    azet[:] = np.nan
    dipet[:] = np.nan
    xmt[:] = np.nan
    ymt [:]= np.nan
    zmt [:]= np.nan
    xbt[:] = np.nan
    ybt[:] = np.nan
    zbt[:] = np.nan
    xet[:] = np.nan
    yet[:] = np.nan
    zet [:]= np.nan
    
    fieldnames=['Company_ID','CollarID','FromDepth','ToDepth','Company_LithoCode','Company_Litho', 'CET_Litho','Score','Level_3', 'Level_2','Level_1',
               'xbt','ybt','zbt','xmt','ymt', 'zmt', 'xet','yet','zet']
    out= open(DB_Lithology_Export_Calc, "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')

    indbt = 0
    indet = 0
    inds = 0
    indt = 0
    for jc in range(nc):
        indbs = -1
        indes = -1
        for js in range(inds, ns):
            if idc[jc]==ids[js]:
                inds = js
                indbs = js
                break
        for js in range(inds, ns):
            if idc[jc]!=ids[js]:
                break
            else:
                inds = js
                indes = js
        
        if indbs==-1 or indes==-1:
            continue

        azm1  = azs[indbs]
        dip1 = dips[indbs]
        at = 0.
        
        if indbs==indes:
            continue
            
        x =  xc[jc]
        y =  yc[jc]
        z =  zc[jc]

        for jt in range(indt, nt):
            if idc[jc]==idt[jt]:
                indt = jt
                #from
                azm2,dip2 = angleson1dh(indbs,indes,ats,azs,dips,fromt[jt])
                azbt[jt] = azm2
                dipbt[jt] = dip2
                len12 = float(fromt[jt]) - at
                dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                xbt[jt] = de
                ybt[jt] = dn
                zbt[jt] = dz

                 #update
                azm1 = azm2
                dip1 = dip2
                at   = float(fromt[jt])

                #midpoint
                mid = float(fromt[jt]) + float((float(tot[jt])-float(fromt[jt]))/2)
                azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,mid)
                azmt[jt] = azm2
                dipmt[jt]= dip2
                len12 = mid - at
                dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                xmt[jt] = de + xbt[jt]
                ymt[jt] = dn + ybt[jt]
                zmt[jt] = dz + zbt[jt]

                #update
                azm1 = azm2
                dip1 = dip2
                at   = mid

                #to
                azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,float(tot[jt]))
                azet[jt] = azm2
                dipet[jt] = dip2
                len12 = float(tot[jt]) - at
                dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                xet[jt] = de + xmt[jt]
                yet[jt] = dn + ymt[jt]
                zet[jt] = dz + zmt[jt]

                #update
                azm1 = azm2
                dip1 = dip2
                at   = float(tot[jt])

                #calculate coordinates
                xbt[jt] = float(x)+float(xbt[jt])
                ybt[jt] = float(y)+float(ybt[jt])
                zbt[jt] = float(z)+float(zbt[jt])
                xmt[jt] = float(x)+float(xmt[jt])
                ymt[jt] = float(y)+float(ymt[jt])
                zmt[jt] = float(z)+float(zmt[jt])
                xet[jt] = float(x)+float(xet[jt])
                yet[jt] = float(y)+float(yet[jt])
                zet[jt] = float(z)+float(zet[jt])

                # update for next interval
                x = xet[jt]
                y = yet[jt]
                z = zet[jt]
	
                out.write('%s,' %compid[jt])	
                out.write('%s,' %idt[jt])
                out.write('%s,' %fromt[jt])
                out.write('%s,' %tot[jt])
                out.write('%s,' %complc[jt])				
                out.write('%s,' %compl[jt])				
                out.write('%s,' %cetlit[jt])
                out.write('%s,' %score[jt])
                out.write('%s,' %lvl3[jt])				
                out.write('%s,' %lvl2[jt])				
                out.write('%s,' %lvl1[jt])				
                out.write('%s,' %xbt[jt])
                out.write('%s,' %ybt[jt])
                out.write('%s,' %zbt[jt])
                out.write('%s,' %xmt[jt])
                out.write('%s,' %ymt[jt])
                out.write('%s,' %zmt[jt])
                out.write('%s,' %xet[jt])
                out.write('%s,' %yet[jt])
                out.write('%s,' %zet[jt])
                #out.write('%s,' %azbt[jt])
                #out.write('%s,' %dipbt[jt])
                #out.write('%s,' %azmt[jt])
                #out.write('%s,' %dipmt[jt])
                #out.write('%s,' %azet[jt])
                #out.write('%s,' %dipet[jt])
                out.write('\n')
    out.close()
    DB_Lithology_Export_Calculated = pd.read_csv(DB_Lithology_Export_Calc,encoding = "ISO-8859-1")
    DB_Lithology_Export_Calculated.drop_duplicates(subset =["CollarID","FromDepth"], keep = False, inplace = True)
    DB_Lithology_Export_Calculated.to_csv(DB_Lithology_Export_Calc)    
    #seen = set() # set for fast O(1) amortized lookup
    #for line in fileinput.FileInput(DB_Lithology_Export_Calc, inplace=1):
    #    if line in seen: continue # skip duplicate
    #seen.add(line)
    #print (line) # standard output is now redirected to the file
    #rows = csv.reader(open(DB_Lithology_Export_Calc, "rt"))
    #newrows = []
    #for row in rows:
        #if row not in newrows:
            #newrows.append(row)
    #writer = csv.writer(open(DB_Lithology_Export_Calc, "w"))
    #writer.writerows(newrows)
    #modified from pygslib
    
def SavePolydata(polydata, path):
    if not path.lower().endswith('.vtp'):
        path = path + '.vtp'

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()
    #modified from pygslib
    
def intervals2vtk(DB_Lithology_Export_Coordinates, vtkfilename):
    table= pd.read_csv(DB_Lithology_Export_Coordinates,encoding = "ISO-8859-1", dtype='object')
    xb = table['xbt'].values.astype(float)
    yb = table['ybt'].values.astype(float)
    zb = table['zbt'].values.astype(float)
    xe = table['xet'].values.astype(float)
    ye = table['yet'].values.astype(float)
    ze = table['zet'].values.astype(float)

    dlen = xb.shape[0]

    vtkfields={}
    for i in table.columns:
        dtype = table[i].dtype

        if  dtype==np.int8 or dtype==np.int16 or dtype==np.int32 or dtype==np.int64 or dtype==np.float16 or dtype==np.float32 or dtype==np.float64:
            vtkfields[i]= vtk.util.numpy_support.numpy_to_vtk(table[i].values)
            vtkfields[i].SetName(i)
            vtkfields[i].SetNumberOfComponents(1)
        else:
            vtkfields[i]= vtk.vtkStringArray()
            vtkfields[i].SetName(i)
            vtkfields[i].SetNumberOfComponents(1)
            vtkfields[i].SetNumberOfTuples(dlen)
            for l in range(dlen):
                vtkfields[i].SetValue(l,str(table[i][l]))

    points= vtk.vtkPoints()
    npoints = dlen*2
    points.SetNumberOfPoints(npoints)

    line = vtk.vtkLine()
    lines = vtk.vtkCellArray()

    n=-1
    for l in range(dlen):
        n=n+1
        points.SetPoint(n,xb[l], yb[l], zb[l])
        line.GetPointIds().SetId(0,n)
        n=n+1
        points.SetPoint(n,xe[l], ye[l], ze[l])
        line.GetPointIds().SetId(1,n)
        lines.InsertNextCell(line)

    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(points)
    linesPolyData.SetLines(lines)

    for i in vtkfields:
        linesPolyData.GetCellData().AddArray(vtkfields[i])

    SavePolydata(linesPolyData, vtkfilename)
    #modified from pygslib
    
def upscale_litho (DB_Lithology_Export, CET_hierarchy_dico_file, DB_Lithology_Upscaled):
    #Upscaled_Litho = DB_Lithology_Export.set_index ('CET_Litho').join(CET_hierarchy_dico.set_index('Level_3'), rsuffix='2')
    DB_Lithology= pd.read_csv(DB_Lithology_Export,encoding = "ISO-8859-1", dtype='object')
    CET_hierarchy_dico= pd.read_csv(CET_hierarchy_dico_file,encoding = "ISO-8859-1", dtype='object')
    Upscaled_Litho=pd.merge(DB_Lithology, CET_hierarchy_dico, left_on='CET_Litho', right_on='Level_3')
    Upscaled_Litho= Upscaled_Litho.loc[:, ~Upscaled_Litho.columns.str.contains('^Unnamed')]
    #Upscaled_Litho.reset_index(level=0, inplace=True)
    #Upscaled_Litho['CET_Litho']=Upscaled_Litho['index']
    #del Upscaled_Litho['index']
    Upscaled_Litho.to_csv(DB_Lithology_Upscaled)
    
def plot_collar (DB_Collar_Export, geology):
    DB_Collar_Export=pd.read_csv(DB_Collar_Export,encoding = "ISO-8859-1", dtype='object')
    DB_Collar_Export['Longitude']=DB_Collar_Export['Longitude'].astype('float64')
    DB_Collar_Export['Latitude']=DB_Collar_Export['Latitude'].astype('float64')
    DB_Collar_Export=gpd.GeoDataFrame(DB_Collar_Export, geometry=[Point(xy) for xy in zip(DB_Collar_Export.Longitude, DB_Collar_Export.Latitude)])
    #DB_Collar_Export=gpd.GeoDataFrame(DB_Collar_Export, geometry=gpd.points_from_xy(DB_Collar_Export.Longitude, DB_Collar_Export.Latitude))
    geology= gpd.read_file(geology)
    base=geology.plot(column='CODE',figsize=(7,7),edgecolor='#000000',linewidth=0.2)
    plot2 = DB_Collar_Export.plot(ax=base, column='CollarID', color='black', markersize=15)        
    plot2 = plot2.figure; plot2.tight_layout()
	
def addtable(table, table_name):
    """
    Adds a table and assigns a name.

    """

    table_name=table.copy(deep=True) # we remove external reference
    
    table_name['fromdepth'] = table_name['fromdepth'].astype(float)
    table_name['todepth'] = table_name['todepth'].astype(float)
    table_name['collarid'] = table_name['collarid'].astype(str)

    table_name.sort_values(by=['collarid', 'fromdepth'], inplace=True)
    table_name.reset_index(level=None, drop=True, inplace=True, col_level=0, col_fill='')
    return table_name
	
def fillgap1Dhole(in_f,
            in_t,
            id,
            tol=0.01,
            endhole=-1):
    """
    Function to fill gaps in one drillhole.
    """
    
    i=0
    nint=-1
    ngap=-1
    noverlap=-1
    ndat= len(in_f)

    #make a deep copy of the from to intervals and create a memory view
    np_f= np.zeros(ndat, dtype=float)
    np_t= np.zeros(ndat, dtype=float)
    f = np_f
    t = np_t
    
    for i in range(ndat):
        f[i]=in_f[i]
        t[i]=in_t[i]

    #make a long array (reserve memory space)
    np_nf= np.zeros(ndat*2+4, dtype=float)
    np_nt= np.zeros(ndat*2+4, dtype=float)
    np_nID= np.zeros(ndat*2+4, dtype=int)
    np_gap= np.zeros(ndat*2+4, dtype=int)
    np_overlap= np.zeros(ndat*2+4, dtype=int)

    nt = np_nt
    nID = np_nID
    gap = np_gap
    nf = np_nf
    overlap = np_overlap

    # gap first interval
    if f[0]>tol:
        nint+=1
        nf[nint]=0.
        nt[nint]=f[0]
        nID[nint]=-999
        ngap+=1
        gap[ngap]=id[0]
        #print("1:", nID[nint])

    for i in range(ndat-1):
        
        # there is no gap?
        if -tol<=f[i+1]-t[i]<=tol:
            # existing sample
            nint+=1
            nf[nint]=f[i]
            nt[nint]=f[i+1]
            nID[nint]=id[i]
            #print("2:", nID[nint])
            continue

        # there is a gap?
        if f[i+1]-t[i]>=tol:
            # existing sample
            nint+=1
            nf[nint]=f[i]
            nt[nint]=t[i]
            nID[nint]=id[i]
            #gap
            nint+=1
            nf[nint]=t[i]
            nt[nint]=f[i+1]
            nID[nint]=-999
            ngap+=1
            gap[ngap]=id[i]
            #print("3:", nID[nint])
            continue

        # there is overlap?
        if f[i+1]-t[i]<=-tol:
            # the overlap is smaller that the actual sample?
            if f[i+1]>f[i]:
                # existing sample
                nint+=1
                nt[nint]=max(f[i+1],f[i]) # ising max here to avoid negative interval from>to
                nf[nint]=f[i]
                nID[nint]=id[i]
                noverlap+=1
                overlap[noverlap]=id[i]
                #print("4:", nID[nint])
                continue
            # large overlap?
            else:
                #whe discard next interval by making it 0 length interval
                # this will happen only in unsorted array...
                noverlap+=1
                overlap[noverlap]=id[i]
                # update to keep consistency in next loopondh
                nint+=1
                nt[nint]=t[i] # ising max here to avoid negative interval from>to
                nf[nint]=f[i]
                nID[nint]=id[i]
                f[i+1]=t[i+1]
                #print("5:", nID[nint])

    # there are no problems (like a gap or an overlap)
    if (-tol<=f[ndat-1]-t[ndat-2]<=tol) and ndat>1:
        nint+=1
        nt[nint]=t[ndat-1] # ising max here to avoid negative interval from>to
        nf[nint]=f[ndat-1]
        nID[nint]=id[ndat-1]
        #print("6:", nID[nint])
    else:
        # just add the sample (the problem was fixed in the previous sample)
        nint+=1
        nt[nint]=t[ndat-1] # ising max here to avoid negative interval from>to
        nf[nint]=f[ndat-1]
        nID[nint]=id[ndat-1]
        #print("7:", nID[nint])

    # add end of hole
    if endhole>-1:
        # ranee's adding end interval
        if (tol<endhole-t[ndat-1]) and (endhole-t[ndat-1]>-tol):
            try:
				#gap
                nint+=1
                nf[nint]=t[i+1]
                nt[nint]=endhole
                nID[nint]=-999
                ngap+=1
                gap[ngap]=id[i]
                #print("100:", nID[nint])
            except IndexError as error:
                nt[nint]=endhole
			
        # there is an end of hole gap?
        if (tol>endhole-t[ndat-2]) and ndat>1:
            #print("1")
            nint+=1
            nt[nint]=endhole
            nf[nint]=t[ndat-1]
            nID[nint]=-999
            ngap+=1
            gap[ngap]=-888  # this is a gap at end of hole
            #print("8:", nID[nint])
			
        # there is an end of hole overlap?
        if (tol>endhole-t[ndat-2]) and ndat>1:
            #print("2")
            nint+=1
            nt[nint]=endhole
            nf[nint]=t[ndat-1]
            nID[nint]=-999
            noverlap+=1
            overlap[noverlap]=-888 # this is an overlap at end of hole
            #print("9:", nID[nint])

        # there is no gap or overlap, good... then fix small differences
        if (tol<endhole-t[ndat-2]) and (endhole-t[ndat-2]>-tol):
            #print("3")
            #print(endhole)
            #print(t[ndat-2])
            #print(ndat)
            nt[nint]=endhole
            #print(nt[nint])
            #print("10:", nID[nint])

    # make first interval start at zero, if it is != to zero but close to tolerance
    if 0<nf[0]<=tol:
        nf[0]=0
    #print(np_nf[:nint+1],np_nt[:nint+1],np_nID[:nint+1],np_gap[:ngap+1],np_overlap[:noverlap+1])    
    return np_nf[:nint+1],np_nt[:nint+1],np_nID[:nint+1],np_gap[:ngap+1],np_overlap[:noverlap+1]

def add_gaps(table_name,
            new_table_name,
            tol=0.01,
            clean=True,
            endhole=-1):

        """Fills gaps with new FROM-TO intervals."""
        
        table_name.sort_values(by=['collarid', 'fromdepth'], inplace=True)
        table_name.loc[:,'_id0']= np.arange(table_name.shape[0])[:]
        group=table_name.groupby('collarid')

        #add gaps
        BHID=group.groups.keys()
        nnf=[]
        nnt=[]
        nnID=[]
        nnBHID=[]
        nngap= []
        nnoverlap = []
        for i in BHID:
            nf,nt,nID,gap,overlap=fillgap1Dhole(in_f = group.get_group(i)['fromdepth'].values,
                                          in_t = group.get_group(i)['todepth'].values,
                                          id = group.get_group(i)['_id0'].values,
                                          tol=tol,
                                          endhole=endhole)


            nBHID = np.empty([len(nf)], dtype=object, order='C')
            nBHID[:]=i
            nnf+=nf.tolist()
            nnt+=nt.tolist()
            nnID+=nID.tolist()
            nnBHID+=nBHID.tolist()
            nngap+=gap.tolist()
            nnoverlap+=overlap.tolist()


        #create new table with gaps (only with fields )
        newtable=pd.DataFrame({'collarid':nnBHID, 'fromdepth':nnf,'todepth':nnt,'_id0':nnID})

        newtable=newtable.join(table_name, on='_id0', rsuffix='__tmp__')

        #clean if necessary
        if clean:
            newtable.drop(
               ['collarid__tmp__', 'fromdepth__tmp__','todepth__tmp__','_id0__tmp__'],
               axis=1,inplace=True, errors='ignore')

        #add table to the class
        new_table_name=addtable(newtable,new_table_name)
        return new_table_name
        #return nngap,nnoverlap
		
def min_int(la,
            lb,
            ia,
            ib,
            tol=0.01):
    """
    Given two complete drillholes A, B (no gaps and up to the end of
    the drillhole), this function returns the smaller of two
    intervals la = FromA[ia] lb = FromB[ib] and updates the
    indices ia and ib. There are three possible outcomes

    - FromA[ia] == FromB[ib]+/- tol. Returns mean of FromA[ia], FromB[ib] and ia+1, ib+1
    - FromA[ia] <  FromB[ib]. Returns FromA[ia] and ia+1, ib
    - FromA[ia] >  FromB[ib]. Returns FromB[ia] and ia, ib+1
    """

    # equal ?  
    #if (lb-1)<=la<=(lb+1):
    if la==lb:
        #print("la", la)
        #print("lb", lb)
        #print("1:", (la+lb)/2)
        #print("ia", ia)
        #print("ib", ib)
        #return ia+1, ib+1, (la+lb)/2
        return ia+1, ib+1, la
    
    # la < lb ?
    if la<lb:
        #print("la", la)
        #print("lb", lb)
        #print("2:",  la)
        return ia+1, ib, la

    # lb < la ?
    if lb<la:
        #print("la", la)
        #print("lb", lb)
        #print("3:",  lb)
        return ia, ib+1, lb

def merge_one_dhole(la,lb,
              ida,
              idb,
              tol=0.01):
    """
    Function to merge one drillhole.

    """
    ia=0
    ib=0
    maxia= len (la)
    maxib= len (lb)
    maxiab= len (lb) + len (la)
    inhole = True
    n=-1

    # prepare output as numpy arrays
    np_newida= np.zeros(maxiab, dtype=int)
    np_newidb= np.zeros(maxiab, dtype=int)
    np_lab= np.zeros(maxiab, dtype=float)

    # get memory view of the numpy arrays
    newida = np_newida
    newidb = np_newidb
    lab = np_lab

    #loop on drillhole
    while inhole:
        # get the next l interval and l idex for drillhole a and b
        ia, ib, l = min_int(la[ia], lb[ib], ia, ib, tol=tol)
        #print(ia, ib, l)
        n+=1
        newida[n]=ida[ia-1]
        newidb[n]=idb[ib-1]
        lab[n]=l
        #print(newida[n])
        #print(newidb[n])
        #print(lab[n])

        #this is the end of hole (this fails if maxdepth are not equal)
        if ia==maxia or ib==maxib:
            inhole=False
    #print(n, np_lab[:n+1], np_newida[:n+1], np_newidb[:n+1])
    return n, np_lab[:n+1], np_newida[:n+1], np_newidb[:n+1]

def get_maxdepth(input_table_A,input_table_B):
    """Get maximum depth between tables"""
    #gaps and overlaps
    maxlist=[]
    a= input_table_A['todepth'].unique().tolist()
    maxlist.append(a[-1])
    b= input_table_B['todepth'].unique().tolist()
    maxlist.append(b[-1])
    maxl=float(max(maxlist))
    return maxl


def merge(input_table_A,input_table_B,new_table_name,tol=0.01,clean=True):
    """Combines two tables by intersecting intervals.

    This function requires drillholes without gaps and overlaps.
    You may un add_gaps in table_A and table_B before using
    this function."""
	
    #get maxdepth
    maxl=get_maxdepth(input_table_A,input_table_B)
    
    #fill gaps
    table_A=add_gaps(input_table_A,new_table_name, tol=tol,clean=True, endhole=maxl)
    table_B=add_gaps(input_table_B,new_table_name, tol=tol,clean=True, endhole=maxl)
    #print(table_A)
    #print(table_B)
    
    #gaps and overlaps
    
    table_A.sort_values(by=['collarid', 'fromdepth'], inplace=True)
    table_B.sort_values(by=['collarid', 'fromdepth'], inplace=True)

    # add ID to tables
    table_A.loc[:,'_id0']= np.arange(table_A.shape[0])[:]
    table_B.loc[:,'_id1']= np.arange(table_B.shape[0])[:]

    # create a groups to easily iterate
    groupA=table_A.groupby('collarid')
    #print(groupA)
    groupB=table_B.groupby('collarid')
    #print(groupB)
            
    # prepare fixed long array to send data
    #    input
    np_la=np.empty(table_A.shape[0]+1, dtype = float)
    np_lb=np.empty(table_B.shape[0]+1, dtype = float)
    np_ida=np.empty(table_A.shape[0]+1, dtype = int)
    np_idb=np.empty(table_B.shape[0]+1, dtype = int)

    ll = table_A.shape[0] +  table_B.shape[0] +10
    nBHID = np.empty(ll, dtype=object, order='C')
    
    la = np_la
    lb = np_lb
    ida  = np_ida
    idb  = np_idb
    
    #merge
    tablea= table_A['collarid'].unique().tolist()
    tableb= table_B['collarid'].unique().tolist()
    BHID=list(set(tablea) & set(tableb))

    nnf=[]
    nnt=[]
    nnIDA=[]
    nnIDB=[]
    nnBHID=[]

    keysA= groupA.groups.keys()
    print(keysA)
    keysB= groupB.groups.keys()
    print(keysB)

    for i in BHID:
        # if we really have to merge
        if (i in keysA) and (i in keysB):
            # prepare input data
            # table A drillhole i
            nk=groupA.get_group(i).shape[0]
            for k in range(nk):
                la[k]=groupA.get_group(i)['fromdepth'].values[k]
                ida[k]=groupA.get_group(i)['_id0'].values[k]

            la[nk]=groupA.get_group(i)['todepth'].values[nk-1]
            ida[nk]=groupA.get_group(i)['_id0'].values[nk-1]

            # table B drillhole i
            nj=groupB.get_group(i).shape[0]
            for j in range(nj):
                lb[j]=groupB.get_group(i)['fromdepth'].values[j]
                idb[j]=groupB.get_group(i)['_id1'].values[j]

            lb[nj]=groupB.get_group(i)['todepth'].values[nj-1]
            idb[nj]=groupB.get_group(i)['_id1'].values[nj-1]

            # make sure the two drill holes have the same length
            # by adding a gap at the end of the shortest drillhole
            if lb[nj] > la[nk]:
                nk+=1
                la[nk] = lb[nj]
                ida[nk] = -999
                endf = lb[nj]

            elif la[nk] > lb[nj]:
                nj+=1
                lb[nj] = la[nk]
                idb[nj] = -999
                endf = la[nk]

            # merge drillhole i
            n, np_lab, np_newida, np_newidb = merge_one_dhole(la[:nk+1],lb[:nj+1], ida[:nk+1], idb[:nj+1], tol=0.01)
            #print(n, np_lab, np_newida, np_newidb)

            # dhid
            nBHID[:n]=i
            nnBHID+=nBHID[:n].tolist()
            # from
            nnf+=np_lab[:-1].tolist()
            # to
            nnt+=np_lab[1:].tolist()
            # IDs
            nnIDA+=np_newida[:-1].tolist()
            nnIDB+=np_newidb[:-1].tolist()
            continue

        # it is only on table A?
        if (i in keysA):
            n= groupA.get_group(i).shape[0]
            # in this case we add table A and ignore B
            # dhid
            nBHID[:n]=i
            nnBHID+=nBHID[:n].tolist()
            # from
            nnf+=groupA.get_group(i)['fromdepth'].values.tolist()
            # to
            nnt+=groupA.get_group(i)['todepth'].values.tolist()
            # IDs
            tmp=-999*np.ones(n, dtype='int')
            nnIDA+=groupA.get_group(i)['_id0'].values.tolist()
            nnIDB+= tmp.tolist()
            continue

        # it is only on table B?
        if (i in keysB):

            n= groupB.get_group(i).shape[0]

            # in this case we add table B and ignore A
            # dhid
            nBHID[:n]=i
            nnBHID+=nBHID[:n].tolist()
            # from
            nnf+=groupB.get_group(i)['fromdepth'].values.tolist()
            # to
            nnt+=groupB.get_group(i)['todepth'].values.tolist()
            # IDs
            tmp=-999*np.ones(n, dtype='int')
            nnIDA+= tmp.tolist()
            nnIDB+= groupB.get_group(i)['_id1'].values.tolist()
            continue


    #create new table with intervals and ID
    newtable=pd.DataFrame({'collarid':nnBHID, 'fromdepth':nnf,'todepth':nnt,'_id0':nnIDA,'_id1':nnIDB})
    print(newtable)
	
    # merge with existing data
    newtable=newtable.join(table_A, on='_id0', rsuffix='__tmp__')
    newtable=newtable.join(table_B, on='_id1', rsuffix='__tmp__')
    print(newtable)


    #clean if necessary
    if clean:
        newtable.drop(
            ['collarid__tmp__', 'fromdepth__tmp__','todepth__tmp__','_id0__tmp__','_id1__tmp__'],
            axis=1,inplace=True, errors='ignore')

    #add table to the class
    new_table_name=addtable(newtable,new_table_name)
    new_table_name.to_csv('../data/export/join.csv')