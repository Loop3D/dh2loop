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
from dh2loop import Var
import logging
from logging.handlers import TimedRotatingFileHandler
from DH2_LConfig import host_,port_,DB_,user_,pwd_,export_path,DB_Collar_Rl_Log,DB_Collar_Maxdepth_Log,DB_Survey_Azi_Log,DB_Survey_Dip_Log,DB_Litho_Depth_Log,DB_Litho_Att_Val_Log,worker_proc
import pandas as pd
from multiprocessing import Process,Manager
import math
from collections import Counter
from math import acos, cos, asin, sin, atan2, tan, radians

            
    
            
            
def collar_collar_attri_Final(DB_Collar_Export,src_csr,dst_csr,minlong,maxlong,minlat,maxlat):
    '''
    Function Extracts data from tables collar and collarattr for processing attributes RL and Maxdepth
    Inputs:
        - src_csr : Coordinate Reference System of source 4326
        - dst_csr : Coordinate Reference System of destination 28350 to 28356
        - minlong,maxlong,minlat,maxlat :  coordinates of region 
  
    Output: is a csv file ,the data processed for RL, Maxdepth attribute in required format  
        
    '''

    fieldnames=['CollarID','HoleId','Longitude','Latitude','RL','MaxDepth','X','Y']
    out= open(os.path.join(export_path,DB_Collar_Export), "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    
    logger1 = logging.getLogger('dev1')   #looging data
    logger1.setLevel(logging.INFO)
    DB_Collar_Rl_Log_Name = os.path.join(export_path, DB_Collar_Rl_Log)
    fileHandler1 = logging.FileHandler(DB_Collar_Rl_Log_Name)  #'RL.log')   #DB_Collar_Rl_Log)  #'RL.log')
    logger1.addHandler(fileHandler1)

    logger2 = logging.getLogger('dev2')
    logger2.setLevel(logging.INFO)
    DB_Collar_MD_Log_Name = os.path.join(export_path, DB_Collar_Maxdepth_Log)
    fileHandler2 = logging.FileHandler(DB_Collar_MD_Log_Name)  #'MD.log')    #DB_Collar_Maxdepth_Log)  #'MD.log')
    logger2.addHandler(fileHandler2)
    
    query =""" SELECT collar.id, replace(replace(collar.holeid, '\"', '_'), ',', '_') as holeid, 
		  collar.longitude, collar.latitude, collarattr.attributecolumn, collarattr.attributevalue 
		  FROM public.collar 
		  INNER JOIN collarattr 
		  ON collar.id = collarattr.collarid 
		  WHERE(longitude BETWEEN %s  AND %s AND latitude BETWEEN %s AND %s)
		  ORDER BY collarattr.collarid ASC """
   
    #WHERE(longitude BETWEEN (minlong = COALESCE(%f, minlong)  AND maxlong = COALESCE(%f, maxlong)) AND latitude BETWEEN (minlat = COALESCE(%f, minlat) AND maxlat = COALESCE(%f, maxlat)))
    conn = None
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
    x2=0.0
    y2=0.0
    #create tranformer object with source and destination read from config file
    transformer = Transformer.from_crs(src_csr, dst_csr)
    
   
  
   
    try:
       
       conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
       cur = conn.cursor()
       Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds read from config file
       cur.execute(query,Bounds)
       collar_collarAttr_Filter = [list(elem) for elem in cur]
       DicList_collar_collarattr = [list(elem) for elem in Var.Attr_col_collar_dic_list]
       for collar_ele in collar_collarAttr_Filter:
         #if (collar_ele[0] == 305574):
            #print("its danger")
         for Dic_ele in DicList_collar_collarattr:  # loop through each element of DB extraction
            if(collar_ele[4] == Dic_ele[0]):
               
               if(Dic_ele[1] == 'rl'):  # check for RL
                  #print("1")
                  if(Pre_id== collar_ele[0] or Pre_id ==0 or Cur_id ==collar_ele[0]):
                     #print("2")
                     #list_rl.append(Parse_Num(collar_ele[5]))
                     list_rl.append(Parse_Num_Rl(collar_ele[5],logger1,collar_ele[0]))
                     Pre_id =collar_ele[0]
                     Pre_hole_id = collar_ele[1]
                     Pre_Longitude =collar_ele[2]
                     Pre_latitude = collar_ele[3]
          
                  else:
                     #chk large , with empty case, write old rec to file
                     #print("3")
                     if(len(list_rl)!=0):
                        #print("4")
                        RL = maximum(list_rl,'NAN')
                     else:
                        RL = maximum(list_rl,'NAN')
                        #RL = "NAN"
                     if(len(list_maxdepth)!=0):
                        #print("5")
                        Maxdepth = maximum(list_maxdepth,'NAN')
                     else:
                         Maxdepth = maximum(list_maxdepth,'NAN')
                         #Maxdepth ="NAN"
                         
                     write_to_csv = True
                     
                     x2,y2=transformer.transform(Pre_latitude,Pre_Longitude) # tranform long,latt for x y calculation
                     if(write_to_csv == True):   # write to csv file
                        out.write('%d,' %Pre_id)
                        out.write('%s,' %Pre_hole_id)
                        out.write('%f,' %Pre_Longitude)
                        out.write('%f,' %Pre_latitude)
                        out.write('%s,' %RL)
                        out.write('%s,' %Maxdepth)
                        out.write('%f,' %x2)
                        out.write('%f,' %y2)
                        out.write('\n')
                        write_to_csv =False
                        RL =''
                        Maxdepth =''
                        Pre_id = 0
                        Pre_hole_id = ''
                        Pre_Longitude =0.0
                        Pre_latitude = 0.0
 
                     Cur_id =collar_ele[0]
                     Cur_hole_id = collar_ele[1]
                     Cur_Longitude =collar_ele[2]
                     Cur_latitude = collar_ele[3]

                     list_rl.clear()
                     list_maxdepth.clear()
                     
                     #list_rl.append(Parse_Num(collar_ele[5]))
                     list_rl.append(Parse_Num_Rl(collar_ele[5],logger1,collar_ele[0]))
                     
             
               elif(Dic_ele[1]=='maxdepth'):  # check for maxdepth
                  #print("7")
                  if(Pre_id== collar_ele[0] or Pre_id == 0 or Cur_id ==collar_ele[0] ):
                     #if(collar_ele[5][0] == '-'):
                        #print("7")
                        #list_maxdepth.append(Parse_Num(collar_ele[5])*-1)
                     #else:
                        #print("8")
                        #list_maxdepth.append(Parse_Num(collar_ele[5]))
                        
                      
                     list_maxdepth.append(Parse_Num_Maxdepth(collar_ele[5],logger2,collar_ele[0]))
                     Pre_id =collar_ele[0]
                     Pre_hole_id = collar_ele[1]
                     Pre_Longitude =collar_ele[2]
                     Pre_latitude = collar_ele[3]

               
                  else:
                     if(len(list_rl)!=0):
                        #print("4")
                        RL = maximum(list_rl,'NAN')
                     else:
                        RL = maximum(list_rl,'NAN')
                        #RL ="NAN"
                     if(len(list_maxdepth)!=0):
                        #print("5")
                        Maxdepth = maximum(list_maxdepth,'NAN')
                     else:
                         Maxdepth = maximum(list_maxdepth,'NAN')
                         #Maxdepth = "NAN"


                     write_to_csv = True

                     x2,y2=transformer.transform(Pre_latitude,Pre_Longitude) # tranform long,latt for x y calculation
                     if(write_to_csv == True):   # write to csv file
                        out.write('%d,' %Pre_id)
                        out.write('%s,' %Pre_hole_id)
                        out.write('%f,' %Pre_Longitude)
                        out.write('%f,' %Pre_latitude)
                        out.write('%s,' %RL)
                        out.write('%s,' %Maxdepth)
                        out.write('%f,' %x2)
                        out.write('%f,' %y2)
                        out.write('\n')
                        write_to_csv =False
                        RL =''
                        Maxdepth =''
                        Pre_id = 0
                        Pre_hole_id = ''
                        Pre_Longitude =0.0
                        Pre_latitude = 0.0
        
                     Cur_id =collar_ele[0]
                     Cur_hole_id = collar_ele[1]
                     Cur_Longitude =collar_ele[2]
                     Cur_latitude = collar_ele[3]

                     list_maxdepth.clear()
                     list_rl.clear()
                     
                     #list_maxdepth.append(Parse_Num(collar_ele[5]))
                     list_maxdepth.append(Parse_Num_Maxdepth(collar_ele[5],logger2,collar_ele[0]))
                     
        
         
         #x2,y2=transformer.transform(Pre_latitude,Pre_Longitude) # tranform long,latt for x y calculation
         #if(write_to_csv == True):   # write to csv file
            #out.write('%d,' %Pre_id)
            #out.write('%s,' %Pre_hole_id)
            #out.write('%f,' %Pre_Longitude)
            #out.write('%f,' %Pre_latitude)
            #out.write('%s,' %RL)
            #out.write('%s,' %Maxdepth)
            #out.write('%f,' %x2)
            #out.write('%f,' %y2)
            #out.write('\n')
            #write_to_csv =False
            #RL =''
            #Maxdepth =''
            #Pre_id = 0
            #Pre_hole_id = ''
            #Pre_Longitude =0.0
            #Pre_latitude = 0.0
            #Cur_id = 0
          

         #else:
            #continue
          
   
       cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
       print(error)
    finally:
       if conn is not None:
          conn.close()

   

def Parse_Num_Maxdepth(s1,logger2,collarID):
   
   s1=s1.lstrip().rstrip()
   if s1.isalpha():
      logger2.info("%d, %s, %s" ,collarID ,s1,"alpha in MaxDepth ,In csv NAN is added")
      return(None)
      
   elif s1 == '-999':
      logger2.info("%d, %s ,%s" ,collarID,s1," MaxDepth is -999,In csv NAN is added")
      return(None)
   elif re.match("^[-+]?[0-9]+$", s1):
       if s1[0] == '-' :
           logger2.info("%d, %s, %s" ,collarID,s1," Maxdepth integer -ve,convert to +ve and add to csv file ")
           return(int(s1) * -1)
       else:
           logger2.info("%d ,%s, %s" ,collarID,s1,"Maxdepth integer +ve,in required status to use directly in csv file ")
           return(int(s1))
   elif re.match("[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?", s1):
      if s1[0] =='-':
         logger2.info("%d, %s ,%s" ,collarID,s1," Maxdepth float -ve,convert to +ve and add to csv file ")
         return(float(s1) * -1)
      else :
         
         logger2.info("%d, %s, %s" ,collarID,s1," Maxdepth float +ve,in required status to use directly in csv file  ")
         return(float(s1))
   


def Parse_Num_Rl(s1,logger1,collarID):
    s1=s1.lstrip().rstrip()
    
    if s1.isalpha():
       logger1.info("%d, %s ,%s" ,collarID,s1,"alpha in RL,In csv file NAN is added",)
       return(None)
    elif re.match("^[-+]?[0-9]+$", s1):
       if int(s1) > 10000 :
           logger1.info("%d, %s ,%s" ,collarID,s1," integer RL > 10000,In csv file NAN is added")
           return(None)
       else :
           logger1.info("%d, %s, %s" ,collarID,s1," integer RL ,in required state to use directly in csv file")
           return(int(s1))
    elif re.match("[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?", s1):
       if float(s1) > 10000.0:
          logger1.info("%d, %s ,%s" ,collarID,s1," float RL  > 10000,In csv file NAN is added")
          return(None)
       else :
          logger1.info("%d, %s ,%s" ,collarID,s1," float RL ,in required state to use directly in csv file")
          return(float(s1))


def Parse_Num(s1):
   s1=s1.lstrip()
   if re.match("^[-+]?[0-9]+$", s1):
      return(int(s1))
   elif re.match("[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?", s1):
      return(float(s1))
   elif s1.isalpha():
      return(None)



def maximum(iterable, default):
  #   '''Like max(), but returns a default value if iterable is empty.'''
    try:
        return str(max(i for i in iterable if i is not None))
    except ValueError:
        return default





def collar_attr_col_dic():
   '''
   Function to extract rl,maxdepth dictionary from DB, and stored in list
   '''
   
   query = '''SELECT  thesaurus_collar_elevation.attributecolumn,thesaurus_collar_elevation.cet_attributecolumn  FROM thesaurus_collar_elevation
              union all 
              SELECT  thesaurus_collar_maxdepth.attributecolumn,thesaurus_collar_maxdepth.cet_attributecolumn  FROM thesaurus_collar_maxdepth'''       

   conn = None
   
   try:
      
      conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
      cur = conn.cursor()
      cur.execute(query)

      for rec in cur:
         Var.Attr_col_collar_dic_list.append(rec)

   
      #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)   # for testing 
   
      #with open('Dic_attr_col_collar.csv', 'w',encoding="utf-8") as f:
         #cur.copy_expert(outputquery, f)
      
 
      cur.close()
   except (Exception, psycopg2.DatabaseError) as error:
      print(error)
   finally:
      if conn is not None:
         conn.close()

   
         

def Survey_Final(DB_Survey_Export,minlong,maxlong,minlat,maxlat):
   '''
   Function which extracts data from tables dhsurvey,dhsurveyattr and collar  for attributes Depth,Azimuth and Dip
   Inputs:
        - minlong,maxlong,minlat,maxlat :  coordinates of region 
   Output:
        - DB_Survey_Export : The processed data after extraction is written to this csv file in required format.
   '''
      
   logger1 = logging.getLogger('dev1')
   logger1.setLevel(logging.INFO)
   DB_Survey_Dip_Log_Name = os.path.join(export_path, DB_Survey_Dip_Log)
   fileHandler1 = logging.FileHandler(DB_Survey_Dip_Log_Name)
   logger1.addHandler(fileHandler1)


   logger2 = logging.getLogger('dev2')
   logger2.setLevel(logging.INFO)
   DB_Survey_Azi_Log_Name = os.path.join(export_path, DB_Survey_Azi_Log)
   fileHandler2 = logging.FileHandler(DB_Survey_Azi_Log_Name)
   logger2.addHandler(fileHandler2)
   
   fieldnames=['CollarID','Depth','Azimuth','Dip']
   out= open(os.path.join(export_path,DB_Survey_Export), "w",encoding ="utf-8")
   for ele in fieldnames:
        out.write('%s,' %ele)
   out.write('\n')
   query =""" select t1.collarid,t1.depth,t2.attributecolumn,t2.attributevalue,t2.dhsurveyid  
		from public.dhsurvey t1
		inner join public.collar 
		on collar.id = t1.collarid
		inner join dhsurveyattr t2
		on t1.id = t2.dhsurveyid
		where((collar.longitude BETWEEN %s AND %s) AND(collar.latitude BETWEEN %s AND %s) )
		order by collar.id ASC """
   conn = None
   AZI = 0.0
   AZI_list =0.0
   AZI_sub_list=[]
   AZI_DIP_LIST =[]
   AZI_ele = 0.0
   DIP = -90 #default Dip to -90
   Pre_id =0
   b_AZI =False
   b_DIP =False
   b_DEPTH =False
   back_survey_0 =0
   back_survey_1 = -1.1
   One_DIP=False
   One_AZI =False
   
   
   try:
      
      conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
      cur = conn.cursor()
      Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
      cur.execute(query,Bounds)
      Survey_First_Filter = [list(elem) for elem in cur]
      Survey_dic_list = [list(elem) for elem in Var.Attr_col_survey_dic_list] 
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
                        
                     #if Pre_id == 125476 :
                        #print("125476")
                        #print(AZI_DIP_LIST)


                     AZI_1 =0.0
                     AZI_2 =0.0
                     DIP_1 =0.0
                     DIP_2 =0.0
                     for loop1_ele in AZI_DIP_LIST:
                        for loop2_ele in AZI_DIP_LIST:
                           if(loop1_ele[0] == loop2_ele[0]):

                                 if abs(loop1_ele[1]) == abs(loop2_ele[1]) and abs(loop1_ele[2]) == abs(loop2_ele[2]):
                                    AZI_1=loop1_ele[1]
                                    DIP_1 = loop1_ele[2]
                                    
                                 elif abs(loop1_ele[1]) != abs(loop2_ele[1]) and abs(loop1_ele[2]) != abs(loop2_ele[2]):
                                    if abs(loop1_ele[1]) > abs(loop2_ele[1]):
                                       AZI_2 = loop1_ele[1]
                                     
                                    else:
                                       AZI_2 = loop2_ele[1]
                                     
                                    
                                    if abs(loop1_ele[2]) > abs(loop2_ele[2]):
                                       if(abs(loop1_ele[2]) ==90):  #default DIP bug solved
                                          DIP_2 = loop2_ele[2]
                                       else:
                                          DIP_2 = loop1_ele[2]
                                      
                                    else:
                                       if(abs(loop2_ele[2]) ==90): #default DIP bug solved
                                          DIP_2 = loop1_ele[2]
                                       else:
                                          DIP_2 = loop2_ele[2]

                                    #if(abs(loop1_ele[2]) ) == 90 :
                                       #DIP_2 = loop2_ele[2]
                                    #elif(abs(loop2_ele[2]) ) == 90 :
                                       #DIP_2 = loop1_ele[2] 
                                   

                                   
                        if abs(AZI_1) > abs(AZI_2):
                           AZI_ = AZI_1
                        else:
                            AZI_ = AZI_2

                        if abs(DIP_1) > abs(DIP_2):
                           if(abs(DIP_1 ) ==90): #default DIP bug solved
                              DIP_ = DIP_2
                           else:
                              DIP_ = DIP_1
                        else:
                           if(abs(DIP_2) ==90): #default DIP bug solved
                              DIP_ = DIP_1
                           else :
                              DIP_ = DIP_2

                            
                        
                        AZI_DIP_Print.append([loop1_ele[0],AZI_,DIP_])
                        AZI_1 =0.0
                        AZI_2 =0.0
                        DIP_1 =0.0
                        DIP_2 =0.0
                        AZI_= 0.0
                        DIP_ = 0.0
                           
   
                     #if Pre_id ==125476  :   #1914687
                        #print(AZI_DIP_Print)
                     
                     b_set = set(tuple(x) for x in AZI_DIP_Print)
                     AZI_DIP_Print_Filter = [ list(x) for x in b_set ]

                     #if Pre_id == 125476 :
                        #print(AZI_DIP_Print_Filter)

                     AZI_DIP_Print_Filter = dict((x[0], x) for x in AZI_DIP_Print_Filter).values()

                 

                     One_AZI= False

                     #if Pre_id == 117689:
                        #print(AZI_DIP_Print_Filter_ele[0])
                        #print(AZI_DIP_Print_Filter_ele[1])
                        #print(AZI_DIP_Print_Filter_ele[2])
                        #print(One_AZI)
                     #print(AZI_DIP_Print_Filter)
                     df = pd.DataFrame(AZI_DIP_Print_Filter,columns=['Depth','Azimuth','Dip'])
                     df.sort_values("Depth", axis = 0, ascending = True, inplace = True)
                     AZI_DIP_Print_Filter = df.values.tolist()
                     if(len(AZI_DIP_Print_Filter)!=0):
                        for AZI_DIP_Print_Filter_ele in AZI_DIP_Print_Filter:
        
                           out.write('%d,' %back_survey_0)
                           out.write('%d,' %AZI_DIP_Print_Filter_ele[0])
                           out.write('%f,' %AZI_DIP_Print_Filter_ele[1])
                           out.write('%f,' %AZI_DIP_Print_Filter_ele[2])
                           out.write('\n')
                           

                           #if Pre_id == 117689:
                              #print(AZI_DIP_Print_Filter_ele[0])
                             # print(AZI_DIP_Print_Filter_ele[1])
                              #print(AZI_DIP_Print_Filter_ele[2])
                             # print(One_AZI)
                     
                     AZI_DIP_Print.clear()
                     
                      
                  AZI_DIP_LIST.clear()
                  
                  if(One_AZI==True):
                     out.write('%d,' %back_survey_0)
                     out.write('%d,' %back_survey_1)
                     out.write('%f,' %AZI)
                     out.write('%f,' %DIP)
                     out.write('\n')
                  AZI =0.0
                  DIP =-90  #default Dip to -90
                  #One_DIP =False
                  One_AZI =False
                  AZI_sub_list.clear()
                  AZI_ele =0.0
 
                  back_survey_0 = 0
                  back_survey_1 = -1.1
                  Pre_id =0


                     
               if ('AZI' in attr_col_ele[1] and (Pre_id ==0 or Pre_id ==survey_ele[0])): # and back_survey_1 == survey_ele[1] ):   #AZI  processing
                  Pre_id = survey_ele[0]
                  if survey_ele[3].isalpha():
                     logger2.info("%d, %d ,%s" ,survey_ele[4],survey_ele[0]," Azi Alpha , It is not considered")
                     continue
                  elif survey_ele[3].replace('.','',1).lstrip('-').isdigit():
                     logger2.info("%d, %d ,%s" ,survey_ele[4],survey_ele[0]," Azi is -ve ,Sign is removed then considered.")
                     if float((survey_ele[3]).replace('\'','').replace('>','').replace('<','').strip())  > 360:
                        logger2.info("%d, %s ,%s" ,survey_ele[4],survey_ele[0]," Azi is > 360 , It is not considered")
                        continue
                     else:
                        logger2.info("%d, %d ,%s" ,survey_ele[4],survey_ele[0]," Azi is valid , It is considered")
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
                           DIP=-90 #default Dip to -90
                           AZI = float((survey_ele[3]).replace('\'','').strip().rstrip('\n\r'))
                           AZI_sub_list.append(AZI)
                           back_survey_0 =survey_ele[0]
                           back_survey_1 = survey_ele[1]
                           One_AZI =False
                           
                           
                           

               if ('DIP' in attr_col_ele[1] and (Pre_id ==survey_ele[0] or Pre_id ==0)) :   #DIP  processing
                  Pre_id = survey_ele[0]
                  if survey_ele[3].isalpha():
                     logger1.info("%d, %d ,%s" ,survey_ele[4],survey_ele[0]," Dip is Alpha , It is not considered")
                     continue
                  elif survey_ele[3].replace('.','',1).lstrip('-').isdigit():
                     if float((survey_ele[3]).replace('\'','').replace('<','').strip())  > 90:  # combine al skip cases
                        logger1.info("%d, %d ,%s" ,survey_ele[4],survey_ele[0]," Dip is > 90 , It is not considered")
                        continue
                     elif float((survey_ele[3]).replace('\'','').replace('<','').strip()) < 0 or float((survey_ele[3]).replace('\'','').replace('<','').strip()) == 0 :
                        logger1.info("%d, %d ,%s" ,survey_ele[4],survey_ele[0]," Dip is <= 0 , It is considered")
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
                           DIP=-90  #default Dip to -90
                           AZI=0.0
                           DIP= float((survey_ele[3]).replace('\'','').replace('<','').replace('>','').strip())
                    
                           back_survey_0 =survey_ele[0]
                           back_survey_1 = survey_ele[1]
        
                           
                        
                 
                        
               #if (Pre_id ==survey_ele[0] or Pre_id ==0):  # depth # chk all corrections
                  #Pre_id = survey_ele[0]
                  #if float(survey_ele[1])
                    # survey_ele[1] = abs(survey_ele[1])
                  #b_DEPTH =True
                 # back_survey_0 =survey_ele[0]
                 # back_survey_1 = survey_ele[1]
                  
                  
   
      cur.close()
   except (Exception, psycopg2.DatabaseError) as error:
      print(error)
   finally:
      if conn is not None:
         conn.close()

   




def Attr_col_dic():
   '''
   Function extracts survey dictionary for attribute column AZI, Dip from DB and stores in List
   '''
   
   
   query = ''' SELECT  thesaurus_survey_azimuth.attributecolumn,thesaurus_survey_azimuth.cet_attributecolumn  FROM thesaurus_survey_azimuth
               union all 
               SELECT  thesaurus_survey_dip.attributecolumn,thesaurus_survey_dip.cet_attributecolumn  FROM thesaurus_survey_dip   '''

   conn = None
   temp_list =[]
   try:
      
      conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
      cur = conn.cursor()
      cur.execute(query)

      for rec in cur:
         Var.Attr_col_survey_dic_list.append(rec)

         
      #Attr_col_survey_dic_list = [list(elem) for elem in temp_list]

      #for ele in Attr_col_survey_dic_list:
         #print(ele)
         #Attr_col_survey_dic_list.append(record)

      #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
      #with open('Dic_attr_col_survey.csv', 'w') as f:
         #cur.copy_expert(outputquery, f)
      
 
          
 
      cur.close()
   except (Exception, psycopg2.DatabaseError) as error:
      print(error)
   finally:
      if conn is not None:
         conn.close()

   
         




def count_Digit(n):
    if n > 0:
        digits = int(math.log10(n))+1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n))+1 # +1 if you don't count the '-'
  
    return digits


def convert_survey(DB_Collar_Export,DB_Survey_Export,DB_Survey_Export_Calc):
   '''
   Function takes collar and survey extracted information and calculates X,Y,Z
   Input :
        - DB_Collar_Export: Data extracted and processed from collar and related tables
        - DB_Survey_Export: Data extracted and processed from survey and related tables
   Output:
        - DB_Survey_Export_Calc :x,y,z calculations for survey data 
   '''
   
   location=pd.read_csv(export_path +'/' +DB_Collar_Export)
   survey=pd.read_csv(export_path +'/' +DB_Survey_Export)
   survey=pd.merge(survey,location, how='left', on='CollarID')

   fieldnames=['CollarID','Depth','Azimuth','Dip','X','Y','Z']
   out= open(os.path.join(export_path,DB_Survey_Export_Calc), "w",encoding ="utf-8")
   for ele in fieldnames:
      out.write('%s,' %ele)
   out.write('\n')
	
   last_CollarID= ''
   for index,row in survey.iterrows():
      if(row['CollarID'] != last_CollarID):
         X1=0.0
         Y1=0.0
         Z1=0.0
         last_Dip =0.0
         last_Azi =0.0
         last_Depth =0.0
         last_CollarID =0.0
         last_Dip=float(row['Dip'])
         last_Azi=float(row['Azimuth'])
         last_Depth=float(row['Depth'])
         last_CollarID=(row['CollarID'])
         X1=float(row['X'])
         Y1=float(row['Y'])
         Z1=float(row['RL'])
      
			
         out.write('%s,' %last_CollarID)
         out.write('%f,' %last_Depth)
         out.write('%f,' %last_Azi)
         out.write('%f,' %last_Dip)
         out.write('%f,' %X1)
         out.write('%f,' %Y1)
         out.write('%f,' %Z1)
         out.write('\n')
         
      else:
         #X2=0.0
         #Y2=0.0
         #Z2=0.0
         #len12 = float(row['Depth']) - last_Depth
         #X2,Y2,Z2=dsmincurb(len12,last_Azi,last_Dip,float(row['Azimuth']),float(row['Dip']))
         X2,Y2,Z2=dia2xyz(X1,Y1,Z1,last_Dip,last_Azi,last_Depth,float(row['Dip']),float(row['Azimuth']),float(row['Depth']))  # x,y z calculation by function dis2xyz
         out.write('%s,' %last_CollarID)
         out.write('%f,' %float(row['Depth']))
         out.write('%f,' %float(row['Azimuth']))
         out.write('%f,' %float(row['Dip']))
         out.write('%f,' %X2)
         out.write('%f,' %Y2)
         out.write('%f,' %Z2)
         out.write('\n')
         X1=X2
         Y1=Y2
         Z1=Z2
         last_Dip=float(row['Dip'])
         last_Azi=float(row['Azimuth'])
         last_Depth=float(row['Depth'])
   out.close()




def dia2xyz(X1,Y1,Z1,I1,Az1,Distance1,I2,Az2,Distance2):
   '''
   Function takes two DIP,AZI,Depth values for X,Y,Z value
   Inputs:
           - X1  : x value fron collar extraction for a particular hole
           - Y1  : y value fron collar extraction for a particular hole
           - Z1  : RL value fron collar extraction for a particular hole
           - I1  : DIP_1 value from survey
           - Az1  : Azi_1 value from survey
           - Distance1 : Depth_1 value from survey
           - I2   : DIP_2 value from survey
           - Az2  : Azi_2 value from survey
           - Distance2 :  Depth_2 value from survey
           
   Output:
           - X,Y,Z value for Deppth_1 to Depth_2
        
   '''
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
	
 
   return X2,Y2,Z2





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
        Var.Attr_val_Dic.append(record)
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
        Var.Litho_dico.append(record)
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
        Var.cleanup_dic_list.append(record)
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
	
    for cleanup_dic_ele in Var.cleanup_dic_list:
        cleaned_item =str(cleanup_dic_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
        text = text.replace('cleaned_item','')
    return text











#labelEncoder = LabelEncoder()
#one_enc = OneHotEncoder()
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
    out= open(os.path.join(export_path,"Attr_val_fuzzy.csv"), "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Attr_val_Dic_new = [list(elem) for elem in Var.Attr_val_Dic]
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
        for Litho_dico_ele in Var.Litho_dico:
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
            
            Var.Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
            
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
            
            Var.Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,'Other',bestmatch])  #top_new[1]])
            
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
            
            



def Depth_validation(row_2,row_3,collarid,dhsurveyid,logger1):
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
            logger1.info("%d, %d ,%d, %d ,%s " ,collarid,dhsurveyid,from_depth,to_depth,"todepth is NULL/None ,0.1 added to from_depth result is todepth")
            return from_depth,to_depth
        elif to_depth>from_depth:
            logger1.info("%d, %d ,%d, %d ,%s " ,collarid,dhsurveyid,from_depth,to_depth,"to_depth > from_depth , which is considered as is")
            return row_2,row_3
        elif from_depth == to_depth:
            to_depth = to_depth+0.01
            row_3=to_depth
            logger1.info("%d, %d ,%d, %d ,%s " ,collarid,dhsurveyid,from_depth,to_depth,"to_depth == from_depth , 0.01 is added to to_depth")
            return row_2,row_3
        elif from_depth >to_depth:
            row_2=to_depth       
            row_3=from_depth
            logger1.info("%d, %d ,%d, %d ,%s " ,collarid,dhsurveyid,from_depth,to_depth,"from_depth > to_depth , depths are swapped")
            return row_2,row_3
        
            
            
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


def Final_Lithology(DB_Lithology_Export,minlong,maxlong,minlat,maxlat):
    '''
    Function Extracts data from tables dhgeologyattr,dhgeology,collar,clbody and attribute column lithology table from DB for the specified region.
    For Each row extracted the from and to depth values are validated , generated fuzzywuzzy values for the lithology along with the score are printed .
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - csv file with the extracted data with fuzzywuzzy and score.
    '''
    query = """select t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn, t1.attributevalue,t1.dhgeologyid 
		 from public.dhgeologyattr t1 
		 inner join public.dhgeology t2 
		 on t1.dhgeologyid = t2.id 
		 inner join collar t3 
		 on t3.id = t2.collarid 
		 inner join clbody t4 
		 on t4.companyid = t3.companyid
		 inner join public.thesaurus_geology_lithology t5
		 on t1.attributecolumn = t5.attributecolumn
		 WHERE(t3.longitude BETWEEN %s AND %s) AND(t3.latitude BETWEEN %s AND %s) 
		 ORDER BY t3.companyid ASC"""

    
    logger1 = logging.getLogger('dev1')
    logger1.setLevel(logging.INFO)
    DB_Litho_Depth_Log_Name = os.path.join(export_path, DB_Litho_Depth_Log)
    fileHandler1 = logging.FileHandler(DB_Litho_Depth_Log_Name)
    logger1.addHandler(fileHandler1)
    
    
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    #cur.execute(query)
    Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query,Bounds)
    First_Filter_list = [list(elem) for elem in cur]
    
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Attribute_column','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']  # for looging
    #fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']
    out= open(os.path.join(export_path,DB_Lithology_Export), "w",encoding ="utf-8")
    
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    
    for First_filter_ele in First_Filter_list:
        for Attr_val_fuzzy_ele in Var.Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                #print(Attr_val_fuzzy_ele[0],"\t",Attr_val_fuzzy_ele[1])
                #print(First_filter_ele[0],"\t",First_filter_ele[5])
                First_filter_ele[2],First_filter_ele[3] =Depth_validation(First_filter_ele[2],First_filter_ele[3],First_filter_ele[1],First_filter_ele[6],logger1)
                out.write('%d,' %First_filter_ele[0])
                out.write('%d,' %First_filter_ele[1])
                out.write('%d,' %First_filter_ele[2])
                out.write('%s,' %First_filter_ele[3])
                out.write('%s,' %First_filter_ele[4])  # for logging 
                out.write('%s,' %Attr_val_fuzzy_ele[1])
                out.write('%s,' %Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',',''))
                out.write('%s,' %Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',',''))   #.replace(',' , ''))
                out.write('%d,' %int(Attr_val_fuzzy_ele[5]))
                out.write('\n')

    
        #for column in First_filter_ele:
            #out_first_filter.write('%s,' %column)
        #out_first_filter.write('\n')
        	
Attr_col_list =[]	
First_Filter_list =[]
Litho_dico=[]


def Attr_COl():
    #query = """SELECT * FROM public.dic_att_col_lithology"""
    query = """SELECT * FROM public.dic_att_col_lithology_1"""  # logging
    #conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        Attr_col_list.append(record)
    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
    with open('Dic_attr_col.csv', 'w') as f:
        cur.copy_expert(outputquery, f)
    

    cur.close()
    conn.close()
    
    
    
    
def First_Filter():
    print("------------------start First_Filter------------")
    start = time.time()
    #out= open("DB_lithology_First1.csv", "w",encoding ="utf-8")
    query = """select t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn, t1.attributevalue ,t1.dhgeologyid 
    from public.dhgeologyattr t1 
    inner join public.dhgeology t2 
    on t1.dhgeologyid = t2.id 
    inner join collar t3 
    on t3.id = t2.collarid 
    inner join clbody t4 
    on t4.companyid = t3.companyid 
    WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN - 30.5 AND - 27.5) 
    ORDER BY t3.companyid ASC"""


    #conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
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
                        #print(row)
                    elif from_depth == to_depth:
                        to_depth = to_depth+0.01
                        row[3]=to_depth
                        First_Filter_list.append(row)
                        #print(row)
                    elif from_depth >to_depth:   
                        row[2]=to_depth       
                        row[3]=from_depth
                        First_Filter_list.append(row)
                        #print(row)
                 
                    #for column in row:
                        #out.write('%s,' %column)
                    #out.write('\n')
                   
                    
   

    cur.close()
    conn.close()
    #out.close() 
    end = time.time()
    #print(end - start)
    
    
def Final_Lithology_old():
    print("--------start of Final -----------")
    bestmatch=-1
    bestlitho=''
    top=[]
    #p = re.compile(r'[- _]')
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','cleaned_text','CET_Lithology','Score']
    out= open("DB_lithology_Final_old.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')

    #with open('Att_Val.log', 'w'):   # to clear the log files 
    #    pass

    logger1 = logging.getLogger('dev1')
    logger1.setLevel(logging.INFO)
    DB_Litho_Att_Val_Log_Name = os.path.join(export_path, DB_Litho_Att_Val_Log)
    fileHandler1 = logging.FileHandler(DB_Litho_Att_Val_Log_Name)
    logger1.addHandler(fileHandler1)

    query = '''SELECT dic_attr_val_lithology_filter.company_id,dic_attr_val_lithology_filter.company_code,replace(dic_attr_val_lithology_filter.comapany_litho, ',' , '_') as comapany_litho  FROM dic_attr_val_lithology_filter'''
    #conn = psycopg2.connect(host='130.95.198.59', port = 5432, database='gswa_dh', user='postgres', password='loopie123pgpw')
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
    a_list = [list(elem) for elem in cur]
    for row in a_list:    
        for First_filter_ele in First_Filter_list:
            #ele_0 = str(First_filter_ele[0]).replace('(','').replace(')','').replace(',','').replace('\'','')    
            #ele_5 = str(First_filter_ele[5]).replace('(','').replace(')','').replace(',','').replace('\'','')
            
            company_code = row[1]
            company_litho = row[2]
            #print(row[0])
            #print( First_filter_ele[0])
            #print(row[1])
            #print( First_filter_ele[5])
            if int(row[0]) == First_filter_ele[0] and  row[1] == First_filter_ele[5]:
                #del First_filter_ele[4]
                #del First_filter_ele[4]



                #cleaned_text=clean_text(row[2])   # without tokenization

                # for logging with tokenization
                
                cleaned_text_1=clean_text(row[2])
                cleaned_text_2=tokenize_and_lemma(cleaned_text_1)
                cleaned_text=" ".join(str(x) for x in cleaned_text_2)
                #logger1.info("%d, %d, %s, %s ,%s" ,First_filter_ele[0],First_filter_ele[6] ,row[2],cleaned_text_1,cleaned_text)  # logging 
                
                
                cleaned_text =  cleaned_text.replace(' rock ',' rocks')   # to handle rock and rocks to get proper fuzzywuzzy
                cleaned_text =  cleaned_text.replace(' rock',' rocks') 
                #print(cleaned_text)
                words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
                #words=words.split(" ")
                words=words.rstrip('\n\r').split(" ")
                last=len(words)-1 #position of last word in phrase

                
                
                for Litho_dico_ele in Litho_dico:              
                    #litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').split(" ")
                    #litho_words=re.split(p, str(Litho_dico_ele))
                    litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').split(" ")
                    scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
                    for sc in scores:                        
                        if(sc[1]>bestmatch): #better than previous best match
                            bestmatch =  sc[1]
                            bestlitho=litho_words[0]
                            #top=sc
                            top.append([sc[0],sc[1]])
                            if(sc[0]==words[last]): #bonus for being last word in phrase
                                bestmatch=bestmatch * 1.01
                        elif (sc[1]==bestmatch): #equal to previous best match
                            if(sc[0]==words[last]): #bonus for being last word in phrase
                                bestlitho=litho_words[0]
                                bestmatch=bestmatch*1.01
                            else:
                                #top=top+sc
                                top.append([sc[0],sc[1]])

                #top = [list(elem) for elem in top]
                #top_new = list(top)
                #if top_new[1] >80:
                if bestmatch >80:
                    #del First_filter_ele[4]
                    #del First_filter_ele[4]
                    #for column in First_filter_ele:
                    out.write('%s,' %First_filter_ele[0])
                    out.write('%s,' %First_filter_ele[1])
                    out.write('%s,' %(First_filter_ele[2]))   #.replace(',' ,' '))
                    out.write('%s,' %First_filter_ele[3])
                    out.write('%s,' %row[1])
                    out.write('%s,' %row[2])
                    out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
                    out.write('%s,' %bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
                    #CET_Litho = str(top_new[0]).replace('(','').replace(')','').replace('\'','').replace(',','')
                    #CET_Litho = CET_Litho.replace(',', ' ')
                    #out.write('%s,' %CET_Litho)
                    out.write('%d,' %bestmatch)    #top_new[1])
                    out.write('\n')
                    logger1.info("%d, %d, %s, %s ,%s , %s , %d " ,First_filter_ele[0],First_filter_ele[6] ,row[2],cleaned_text_1,cleaned_text,bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''),bestmatch)  # logging 
                    top.clear()
                    #top_new[:] =[]
                    CET_Litho=''
                    bestmatch=-1
                    bestlitho=''
                else:
                    #del First_filter_ele[4]
                    #del First_filter_ele[4]
                    #for column in First_filter_ele:
                    out.write('%s,' %First_filter_ele[0])
                    out.write('%s,' %First_filter_ele[1])
                    out.write('%s,' %(First_filter_ele[2]))   #.replace(',' ,' '))
                    out.write('%s,' %First_filter_ele[3])
                    out.write('%s,' %row[1])
                    out.write('%s,' %row[2])
                    out.write('%s,' %cleaned_text)
                    out.write('Other,')
                    out.write('%d,' %bestmatch)   #top_new[1])
                    out.write('\n')
                    logger1.info("%d, %d, %s, %s ,%s ,%s , %d " ,First_filter_ele[0],First_filter_ele[6] ,row[2],cleaned_text_1,cleaned_text,'Other',bestmatch)  # logging 
                    top.clear()
                    #top_new[:] =[]
                    CET_Litho=''
                    bestmatch=-1
                    bestlitho=''

    cur.close()
    conn.close()
    out.close()


def Upscale_lithology(DB_Lithology_Export,DB_Lithology_Upscaled_Export):
    '''
    Function upscales the CET_Loithology generated using the CET hierarchy dictionary to level1,level2,level3
    Input: 
        - DB_Lithology_Export csv file 
    Output:
        - is a csv file DB_Lithology_Upscaled_Export with upscales data 
    '''

    Hierarchy_litho_dico_List =[]
    query = """ select thesaurus_geology_hierarchy.detailed_lithology,thesaurus_geology_hierarchy.lithology_subgroup,thesaurus_geology_hierarchy.lithology_group  
            from thesaurus_geology_hierarchy """
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    cur.execute(query)
    Hierarchy_litho_dico_List  = [list(elem) for elem in cur]
    CET_hierarchy_dico = pd.DataFrame(Hierarchy_litho_dico_List,columns=['detailed_lithology','lithology_subgroup','lithology_group'])
    #CET_hierarchy_dico.to_csv ('CET_hierarchy_dico.csv', index = False, header=True)
    #print (CET_hierarchy_dico)
    DB_Lithology= pd.read_csv(export_path +'/'+ DB_Lithology_Export,encoding = "ISO-8859-1", dtype='object')
    Upscaled_Litho=pd.merge(DB_Lithology, CET_hierarchy_dico, left_on='CET_Lithology', right_on='detailed_lithology')
    Upscaled_Litho.sort_values("Company_ID", ascending = True, inplace = True)
    #Upscaled_Litho.drop(['Unnamed: 8'], axis=1)
    del Upscaled_Litho['Unnamed: 9']
    Upscaled_Litho.to_csv (export_path +'/'+ DB_Lithology_Upscaled_Export, index = False, header=True)
    
    



def Remove_duplicates_Litho(DB_Lithology_Upscaled_Export,Upscaled_Litho_NoDuplicates_Export):
    '''
    Function removes the multiple companies logging the same lithology (or duplicate rows)
    Input:
        - DB_Lithology_Upscaled_Export csv file
    Output:
        - Upscaled_Litho_NoDuplicates_Export csv file.
    '''
    Final_Data= pd.read_csv(export_path +'/' + DB_Lithology_Upscaled_Export)   
    Final_Data.CollarID = Final_Data.CollarID.astype(int)
    Final_Data.Fromdepth = Final_Data.Fromdepth.astype(float)
    Final_Data.Todepth = Final_Data.Todepth.astype(float)
    Final_Data.sort_values(['CollarID', 'Fromdepth','Todepth'], inplace=True)
    singles = Final_Data.drop_duplicates(subset=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode'],keep='first',inplace =False)   #,'Company_Lithology','CET_Lithology','Score'
    singles.to_csv(export_path +'/'+ Upscaled_Litho_NoDuplicates_Export,index=False)





def dsmincurb (len12,azm1,dip1,azm2,dip2):
    #DEG2RAD = 3.141592654/180.0
    #i1 = (90 - float(dip1)) * DEG2RAD
    i1 = np.deg2rad(90 - float(dip1))
    #a1 = float(azm1) * DEG2RAD
    a1 = np.deg2rad(float(azm1))
    #i2 = (90 - float(dip2)) * DEG2RAD
    i2 = np.deg2rad(90 - float(dip2))
    #a2 = float(azm2) * DEG2RAD
    a2 = np.deg2rad(float(azm2))  #DEG2RAD
	
    #Beta = acos(cos(I2 - I1) - (sin(I1)*sin(I2)*(1-cos(Az2-Az1))))
    dl = acos(cos(float(i2)-float(i1))-(sin(float(i1))*sin(float(i2))*(1-cos(float(a2)-float(a1)))))
    if dl!=0.:
        rf = 2*tan(dl/2)/dl  # minimum curvature
    else:
        rf=1				 # balanced tangential
    dz = 0.5*len12*(cos(float(i1))+cos(float(i2)))*rf
    dn = 0.5*len12*(sin(float(i1))*cos(float(a1))+sin(float(i2))*cos(float(a2)))*rf
    de = 0.5*len12*(sin(float(i1))*sin(float(a1))+sin(float(i2))*sin(float(a2)))*rf
    return dz,dn,de
	#modified from pygslib

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
    #DEG2RAD=3.141592654/180.0
    # convert degree to rad and correct sign of dip
    #razm = float(azm) * float(DEG2RAD)
    razm =  float(np.deg2rad(float(azm)))
    #rdip = -(float(dip)) * float(DEG2RAD)
    rdip =  float(np.deg2rad(-float(dip)))

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
    #RAD2DEG=180.0/3.141592654
    #pi = 3.141592654
    azm= float(atan2(x,y))
    if azm<0.:
        azm= azm + math.pi*2
    #azm= azm + math.pi*2
    #azm = float(azm) * float(RAD2DEG)
    azm =float(np.rad2deg(float(azm)))
    #dip = -(float(asin(z))) * float(RAD2DEG)
    dip =-float(np.rad2deg(float(asin(z))))
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

def Convert_lithology(DB_Collar_Export, DB_Survey_Export, Upscaled_Litho_NoDuplicates_Export, DB_Lithology_Export_Calc):
    print("--------start of convert Lithology -----------")
    
    #collar= pd.read_csv('DB_Collar_Export.csv',encoding = "ISO-8859-1", dtype='object')
    #survey= pd.read_csv('DB_Survey_Export.csv',encoding = "ISO-8859-1", dtype='object')
    #litho= pd.read_csv('Upscaled_Litho_NoDuplicates_Export.csv',encoding = "ISO-8859-1", dtype='object')
    
        
    collar= pd.read_csv(export_path +'/' + DB_Collar_Export )  
    survey= pd.read_csv(export_path +'/' + DB_Survey_Export ) 
    litho= pd.read_csv(export_path +'/' + Upscaled_Litho_NoDuplicates_Export) 
    
    collar.CollarID = collar.CollarID.astype(str)
    survey.CollarID = survey.CollarID.astype(str)
    survey.Depth = survey.Depth.astype(float)
    litho.CollarID = litho.CollarID.astype(str)
    litho.Fromdepth = litho.Fromdepth.astype(float)
    litho.Todepth = litho.Todepth.astype(float)

    #collar.sort_values(['CollarID'], inplace=True)
    #survey.sort_values(['CollarID', 'Depth'], inplace=True)
    #litho.sort_values(['CollarID', 'Fromdepth'], inplace=True)


    global idc
    global xc
    global yc
    global zc
    global idc
    global ats
    global azs
    global dips
    global idt
    global fromt
    global tot
    global cetlit

    
    
    idc = collar['CollarID'].values
    xc = collar['X'].values
    yc = collar['Y'].values
    zc = collar['RL'].values
    ids = survey['CollarID'].values
    ats = survey['Depth'].values
    azs = survey['Azimuth'].values
    dips = survey['Dip'].values
    idt =litho['CollarID'].values
    fromt = litho['Fromdepth'].values
    tot = litho['Todepth'].values
    cetlit=litho['CET_Lithology'].values

    nc= idc.shape[0]
    ns= ids.shape[0]
    nt= idt.shape[0]


    global azmt
    global dipmt
    global xmt
    global ymt
    global zmt
    global azbt
    global dipbt
    global xbt
    global ybt
    global zbt
    global azet
    global dipet
    global xet
    global yet
    global zet
    

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

    fieldnames=['CollarID','FromDepth','ToDepth','Lithology','xbt','ybt','zbt','xmt','ymt', 'zmt', 'xet','yet','zet','azbt','dipbt']
    global out
    out= open(os.path.join(export_path,DB_Lithology_Export_Calc), "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')

    global indbt 
    global indet 
    global inds 
    global indt
    global indbs
    global indes
    global sub_indbs
    global sub_indes
    
    indbt = 0
    indet = 0
    inds = 0
    indt = 0
    ii = 0
    global jc
    global js
    global jt

    global from_depth
    global mid_depth
    global to_depth

    global begin_f 
    global mid_f 
    global end_f

    global prev_dict
    global survey_sub_cnt

    
    
    
    for jc in range(nc):
        indbs = -1
        indes = -1
        for js in range(inds, ns):
            if idc[jc]==ids[js]:
                #print(ids[js])
                inds = js
                indbs = js
                global temp_Survey_df
                if(idc[jc] == '150942'):
                    ii =ii + 1
                #temp_Survey_df = (survey.loc[survey['CollarID'].isin([ids[js]])]).copy()
                temp_Survey_df = (survey.loc[survey['CollarID']== ids[js]])
                #start_pos = temp_Survey_df[temp_Survey_df['CollarID']==ids[js]].index.item()
                #print(start_pos)
                survey_sub_cnt=temp_Survey_df.shape[0]
                #print(survey_sub_cnt)
                #print(temp_Survey_df)
                break
        for js in range(inds, ns):
            if idc[jc]!=ids[js]:
                break
            else:
                inds = js
                indes = js
        
        if indbs==-1 or indes==-1:
            continue

        global azm1 
        global dip1 
        global at 

        azm1  = azs[indbs]
        dip1 = dips[indbs]
        at = 0.
        
        if indbs==indes:
            continue

        global x 
        global y 
        global z

        begin_f = False 
        mid_f  = False
        end_f = False

        global litho_sub_cnt
            
        x =  xc[jc]
        y =  yc[jc]
        z =  zc[jc]

        sub_indbs = indbs
        sub_indes = indes

        litho_ind = 0
        global survey_ind
        survey_ind = 0

        litho_sub_cnt_flag = True
        tmp_litho_sub_cnt = 1

        litho_more_ele = False
        start_pos =0

        prev_dict ={"todepth" : -0.0,
                            "x": -0.0,
                            "y": -0.0,
                            "z": -0.0}

        

        for jt in range(indt, nt):
            #if(idc[jc]!=idt[jt]):
                #break
            if idc[jc]==idt[jt]:
                #print(idt[jt])
                #print(idc[jc])
                global temp_litho_df
                #temp_litho_df = (litho.loc[litho['CollarID'].isin([idt[jt]])]).copy()
                if litho_sub_cnt_flag == True :  # to keep track of count till the end of one collarID
                    temp_litho_df = (litho.loc[litho['CollarID']== idt[jt]])
                    #filtered_df = df.loc[df['Symbol'] == 'A99']
                    litho_sub_cnt=temp_litho_df.shape[0]
                    litho_sub_cnt_flag = False

                    #print(temp_Survey_df)
                    #start_pos = temp_Survey_df[temp_Survey_df['Depth']== 0].index.item()
                    
                    start_pos_list = temp_Survey_df[temp_Survey_df['CollarID']== idt[jt]].index.tolist() # fetch start index of survey sub datafframe #.item()
                    start_pos = start_pos_list[0]  # fetch only the first element index
                    #print(start_pos)
                    #print(temp_Survey_df)
                    

                else :
                    tmp_litho_sub_cnt = tmp_litho_sub_cnt + 1
                    
                #if(litho_sub_cnt > survey_sub_cnt)
                    
                #print(litho_sub_cnt)
                #print(temp_litho_df)
                #print(tot)
                #print(idc[jc])

                if(idc[jc] == '125471'):
                    ii =ii + 1

                
                indt = jt


                
                Depth_survey = temp_Survey_df['Depth'].values
                litho_Todepth = temp_litho_df['Todepth'].values

                print(Depth_survey)
                print(litho_Todepth)
                

                from_depth = fromt[jt]
                mid_depth = float(fromt[jt]) + float((float(tot[jt])-float(fromt[jt]))/2)
                to_depth = tot[jt]

                #end_pos = temp_Survey_df[temp_Survey_df['Depth']==tot[jt]].index.item()
                

                begin_f = True
                mid_f = True
                end_f  = True

                

                #print(idt[jt])
               
                #print(tot[jt] in temp_Survey_df.Depth.values)

                if(tot[jt] in temp_Survey_df.Depth.values  and tot[jt] > Depth_survey [survey_ind+1]  ):
                    #tmp_litho_sub_cnt = litho_sub_cnt -1
                   #loc = np.where(Depth_survey [survey_ind] ==tot[jt])
                    #print(tot[jt])

                    end_pos = temp_Survey_df[temp_Survey_df['Depth']==tot[jt]].index.item()

                    #from_depth = fromt[jt]
                    #to_depth = tot[jt]
                    #fromt[jt] = Depth_survey [survey_ind]
                    #print(pos)
                    #print(tot[jt])
                    while (start_pos < end_pos):
                        #print(survey_ind)
                        #print(Depth_survey [survey_ind])
                        fromt[jt] = Depth_survey [survey_ind]
                        #print(fromt[jt])
                        tot[jt] = Depth_survey [survey_ind + 1]
                        #print(tot[jt])
                        calculate_x_y_z()
                        #print_xyz_csv()
                        survey_ind = survey_ind +1
                        start_pos= start_pos + 1
                        #print("calculated")

                    #print_xyz_csv()

                #elif(tot[jt] not in temp_Survey_df.Depth.values and tot[jt] > max(Depth_survey)  and tmp_litho_sub_cnt <= litho_sub_cnt   and  survey_sub_cnt-1 == survey_ind+1 ) : # survey elements are  over for collarID but litho still has values , Eg collarId 1111
                    #calculate_x_y_z()
                    
                    
                #elif(tot[jt] not in temp_Survey_df.Depth.values and tot[jt] > max(Depth_survey)  and tmp_litho_sub_cnt == litho_sub_cnt  and survey_ind != survey_sub_cnt-2 ) :   #litho in last row but servey is not in last row,
                   # if survey_ind != survey_sub_cnt-2:
                    #    survey_ind = survey_ind +1
                     #   calculate_x_y_z()
                        

                elif(tot[jt] not in temp_Survey_df.Depth.values and tot[jt] > max(Depth_survey)):  #and tmp_litho_sub_cnt == litho_sub_cnt ) :  litho last row but survey got many intermediates to cover
                    #print("Max of Survey", "\t",max(Depth_survey))
                    #if survey_sub_cnt-1 == survey_ind+1 and tmp_litho_sub_cnt == litho_sub_cnt :
                        #litho_more_ele = True
                        
                    #if  survey_ind+1 == survey_sub_cnt-1 and tmp_litho_sub_cnt == litho_sub_cnt :
                        #litho_more_ele = True

                    #if(tot[jt] > Depth_survey [survey_ind+1]  and  survey_ind <= survey_sub_cnt-2):
                        #while True:
                            #if Depth_survey [survey_ind] <=  tot[jt]  >= Depth_survey [survey_ind+1] : #and survey_ind <= survey_sub_cnt -2:
                               #survey_ind = survey_ind +1
                            #else:
                                #litho_more_ele == True
                                #break
                    if survey_sub_cnt-1 == survey_ind and tmp_litho_sub_cnt == litho_sub_cnt :   #survey_sub_cnt-1 == survey_ind+1 and tmp_litho_sub_cnt < litho_sub_cnt :
                        calculate_x_y_z()
                        continue
                        
                    if survey_sub_cnt-1 == survey_ind and tmp_litho_sub_cnt < litho_sub_cnt :   #survey_sub_cnt-1 == survey_ind+1 and tmp_litho_sub_cnt < litho_sub_cnt :
                        calculate_x_y_z()
                        if  tmp_litho_sub_cnt < litho_sub_cnt :
                            litho_more_ele = True
                        elif tmp_litho_sub_cnt == litho_sub_cnt :
                            litho_more_ele = False
                            
                        continue
                    
                    if litho_more_ele == True:
                        calculate_x_y_z()
                        if  tmp_litho_sub_cnt < litho_sub_cnt :
                            litho_more_ele = True
                        elif tmp_litho_sub_cnt == litho_sub_cnt :
                            litho_more_ele = False
                        
                        continue

                    if Depth_survey[survey_ind] <= tot[jt] >= Depth_survey[survey_ind + 1] and tmp_litho_sub_cnt <= litho_sub_cnt and survey_ind +1 == survey_sub_cnt-1:  #survey end but litho still exists Eg: 132164,132168 on 26/8/20
                        survey_ind = survey_ind +1
                        calculate_x_y_z()
                        continue
                        
                    if  Depth_survey[survey_ind] <= tot[jt] >= Depth_survey[survey_ind + 1] and tmp_litho_sub_cnt == litho_sub_cnt and survey_ind +1 == survey_sub_cnt-2:    # survey needs increment Eg hole3 
                        survey_ind = survey_ind +1
                        calculate_x_y_z()
                        continue
                        
                    
                        
                        
                    #tmp_litho_sub_cnt = litho_sub_cnt -1
                    #print(tot[jt])
                    #print(jt)
                    #print(survey_ind)
                    #last_Litho_ToDepth = tot[jt]
                    while(survey_ind <  survey_sub_cnt):
                       #print(survey_ind)
                       #last_Litho_ToDepth = tot[jt]
                       #print(last_Litho_ToDepth)
                       #fromt[jt] = Depth_survey [survey_ind]
                       #print(fromt[jt])
                       #if(survey_ind < survey_sub_cnt-1):
                       if(survey_ind == survey_sub_cnt-1):
                           #print("inside")
                           #print(survey_ind)
                           #print(survey_sub_cnt-2)
                           #print(last_Litho_ToDepth)
                           #tot[jt] = last_Litho_ToDepth

                           if litho_sub_cnt == 1 :   # for holeid = 132171, 1/9/20
                               fromt[jt] = from_depth
                               tot[jt] = to_depth
                               calculate_x_y_z()
                               break



                           
                           fromt[jt] = Depth_survey [survey_ind]
                           #print(fromt[jt])
                           tot[jt] = to_depth
                           #print(tot[jt])
                           calculate_x_y_z()
                           #survey_ind = survey_ind +1
                           break
                           #print_xyz_csv()
                           #survey_ind = survey_ind +1
                           
                       else:
                           #print(survey_ind)
                           fromt[jt] = Depth_survey [survey_ind]
                           #print(fromt[jt])
                           tot[jt] = Depth_survey [survey_ind + 1]
                           #print(tot[jt])
                           calculate_x_y_z()
                           #print_xyz_csv()
                           survey_ind = survey_ind +1
                           #print(survey_ind)

                    #print_xyz_csv()   #chk from to interval in csv


                elif(tot[jt] < Depth_survey [survey_ind+1] or tot[jt] == Depth_survey [survey_ind+1] ):  #and litho_sub_cnt <=0) :
                    calculate_x_y_z()
                    #tmp_litho_sub_cnt = litho_sub_cnt -1
                    #print(tmp_litho_sub_cnt)
                    #print_xyz_csv()


                elif tot[jt] not in temp_Survey_df.Depth.values and tot[jt] > max(Depth_survey) and  tmp_litho_sub_cnt == litho_sub_cnt and end_pos > litho_sub_cnt:   # bug,for holeid 132170
                    end_pos = temp_Survey_df[temp_Survey_df['Depth']==tot[jt]].index.item()
                    while (start_pos < end_pos):
                        #print(survey_ind)
                        #print(Depth_survey [survey_ind])
                        if Depth_survey [survey_ind] <=  tot[jt]  >= Depth_survey [survey_ind+1]:
                            fromt[jt] = Depth_survey [survey_ind]
                            #print(fromt[jt])
                            tot[jt] = Depth_survey [survey_ind + 1]
                            #print(tot[jt])
                            calculate_x_y_z()
                            #print_xyz_csv()
                            survey_ind = survey_ind +1
                            start_pos= start_pos + 1
                        elif Depth_survey [survey_ind] <=  tot[jt]  <= Depth_survey [survey_ind+1]:
                            fromt[jt] = Depth_survey [survey_ind]
                            tot[jt] = to_depth
                            calculate_x_y_z()
                            survey_ind = survey_ind +1
                            start_pos= start_pos + 1
                            
                            
                

                elif(tot[jt] > Depth_survey [survey_ind+1]):
                    while True:
                        if Depth_survey [survey_ind] <=  tot[jt]  >= Depth_survey [survey_ind+1]:
                           survey_ind = survey_ind +1
                           start_pos= start_pos + 1   # todepth needs survey to increment , increment start for condition 1 Eg:collarID 169555
                        else:
                            break
                        
                    #tmp_litho_sub_cnt = litho_sub_cnt -1
                    #survey_ind +1 = survey_ind = survey_ind +2
                    #fromt[jt] = Depth_survey [survey_ind] 
                    #tot[jt] = Depth_survey [survey_ind + 1]
                    calculate_x_y_z()
                    
                    
                    

                
              
                   
                
    out.close()



    

def calculate_x_y_z(): #indbs,indes,ats,azs,dips,fromt,tot,jt):
                #from
                global azm1
                global dip1
                global at

                global xbt 
                global ybt 
                global zbt 
                global xmt 
                global ymt 
                global zmt 
                global xet 
                global yet 
                global zet

                global x 
                global y 
                global z

                global indbs
                global indes

                global begin_f 
                global mid_f 
                global end_f

                global prev_dict
                global litho_sub_cnt
                global survey_sub_cnt
                global survey_ind
                
                
                

                
                    
                if from_depth != fromt[jt] and fromt[jt] <= from_depth <= tot[jt] :   # hole 153637, on 3/9/20
                    fromt[jt] = from_depth
                    azm2,dip2 = angleson1dh(indbs,indes,ats,azs,dips,fromt[jt])
                else:
                    azm2,dip2 = angleson1dh(indbs,indes,ats,azs,dips,fromt[jt])
                    
                
                print(fromt[jt])
                azbt[jt] = azm2
                dipbt[jt] = dip2
                len12 = float(fromt[jt]) - at
                dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                xbt[jt] = de
                ybt[jt] = dn
                zbt[jt] = dz

                #xbt[jt] = float(x)+float(xbt[jt])
                #ybt[jt] = float(y)+float(ybt[jt])
                #zbt[jt] = float(z)+float(zbt[jt])

                
                

                
                
                #print(xbt[jt],"\t",ybt[jt],"\t",zbt[jt])
                #if dipbt[jt] > 0 : # if DIP is +ve, growing UP
                    #zbt[jt] = (dz *-1)
                #else :
                    #zbt[jt] = dz

                 #update
                
                
                azm1 = azm2
                dip1 = dip2
                at   = float(fromt[jt])
                #print(azm1,"\t",dip1,"\t",at)

                #midpoint
                mid = float(fromt[jt]) + float((float(tot[jt])-float(fromt[jt]))/2)
                if (mid_depth == mid or mid_depth == fromt[jt] or mid_depth == tot[jt]) :
                    azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,mid)
                elif ( float(fromt[jt]) <= mid_depth <= float(tot[jt])):
                    mid = mid_depth
                    azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,mid)
                else:
                    azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,mid)
                    
                
                print(mid)
                azmt[jt] = azm2
                dipmt[jt]= dip2
                len12 = mid - at
                dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                xmt[jt] = de + xbt[jt]
                ymt[jt] = dn + ybt[jt]
                zmt[jt] = dz + zbt[jt]

                #xmt[jt] = float(x)+float(xmt[jt])
                #ymt[jt] = float(y)+float(ymt[jt])
                #zmt[jt] = float(z)+float(zmt[jt])

                

                
                    
                    
                
                #print(xmt[jt],"\t",ymt[jt],"\t",zmt[jt])
                #if dipmt[jt] > 0 : # if DIP is +ve, growing UP
                    #zmt[jt] = (dz * -1)+ zbt[jt]
                #else:
                    #zmt[jt] = dz + zbt[jt]

                #update
                azm1 = azm2
                dip1 = dip2
                at   = mid
                #print(azm1,"\t",dip1,"\t",at)

                #to
                azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,float(tot[jt]))
                print(tot[jt])
                azet[jt] = azm2
                dipet[jt] = dip2
                len12 = float(tot[jt]) - at
                dz,dn,de = dsmincurb(len12,azm1,dip1,azm2,dip2)
                xet[jt] = de + xmt[jt]
                yet[jt] = dn + ymt[jt]
                zet[jt] = dz + zmt[jt]

                #xet[jt] = float(x)+float(xet[jt])
                #yet[jt] = float(y)+float(yet[jt])
                #zet[jt] = float(z)+float(zet[jt])

                
                    
                
                #print(xet[jt],"\t",yet[jt],"\t",zet[jt])
                #if dipet[jt] > 0: # if DIP is +ve, growing UP
                    #zet[jt] = (dz * -1) + zmt[jt]
                #else:
                    #zet[jt] = dz + zmt[jt]

                #update
                azm1 = azm2
                dip1 = dip2
                at   = float(tot[jt])
                #print(azm1,"\t",dip1,"\t",at)

                #calculate coordinates
                
                
                xbt[jt] = float(x)+float(xbt[jt])
                ybt[jt] = float(y)+float(ybt[jt])
                zbt[jt] = float(z)+float(zbt[jt])
                #print("begin flag=","\t",begin_f)
                if from_depth == fromt[jt] and begin_f == True :
                    print_xyz_Begin_csv()
                    begin_f = False

                #print(prev_dict)
                #print(begin_f)
                    
                if prev_dict["todepth"] == from_depth and begin_f == True and  survey_ind == survey_sub_cnt-2  :
                    #print(prev_dict)
                    #print(jt)
                    #print(survey_sub_cnt-1)
                    #print(survey_ind)
                    print_xyz_Begin_Prev_csv()
                    begin_f = False

                str_todepth = str(prev_dict["todepth"])   # for holeid = 150934 , on 2/9/20
                before, after = str_todepth.split('.')
                if int(before) == from_depth and begin_f == True :    # for holeid = 150934 , on 2/9/20
                    print_xyz_Begin_Prev_csv()
                    begin_f = False
                    
                
                xmt[jt] = float(x)+float(xmt[jt])
                ymt[jt] = float(y)+float(ymt[jt])
                zmt[jt] = float(z)+float(zmt[jt])
                if mid_depth == mid  and mid_f == True  :
                    print_xyz_Mid_csv()
                    mid_f =False

                #if fromt[jt] <= mid_depth <= mid:
                    #tot[jt] = mid
                    #calculate_x_y_z()
                
                
                #if  mid <= mid_depth <= tot[jt]:
                    #fromt[jt] = mid
                    #calculate_x_y_z()

                

                #if prev_dict["todepth"] == from_depth and mid_f == True and  survey_ind == survey_sub_cnt-2  :
                    #print_xyz_Mid_csv()
                    #mid_f =False

                #if prev_dict["todepth"] == from_depth and mid_f == True and   fromt[jt] <= mid_depth <= mid:     # for last depth who's mid != mid_depth  #survey_ind == survey_sub_cnt-2  :
                    #print_xyz_Mid_csv()
                    #mid_f =False
                    
                    
                xet[jt] = float(x)+float(xet[jt])
                yet[jt] = float(y)+float(yet[jt])
                zet[jt] = float(z)+float(zet[jt])


                #if prev_dict["todepth"] == from_depth and mid_f == True and   mid <= mid_depth <= tot[jt]:     # for last depth who's mid != mid_depth  #survey_ind == survey_sub_cnt-2  :
                    #print_xyz_End_csv()
                    #mid_f =False

                
                if to_depth == tot[jt] and end_f ==True :
                    print_xyz_End_csv()
                    tmp_dict=dict(todepth=tot[jt],x=xet[jt],y=yet[jt],z=zet[jt])
                    prev_dict=tmp_dict
                    #print(prev_dict)
                    end_f =False

                if mid_depth == tot[jt] and mid_f == True :
                    print_xyz_mid_inEnd()
                    mid_f =False

               
                    

                

                print(xbt[jt],"\t",ybt[jt],"\t",zbt[jt])
                print(xmt[jt],"\t",ymt[jt],"\t",zmt[jt])
                print(xet[jt],"\t",yet[jt],"\t",zet[jt])
                #print("survey Index","\t",survey_ind)

                # update for next interval
                
                
                x = xet[jt]
                y = yet[jt]
                z = zet[jt]


def print_xyz_Begin_csv(): #out,idt,fromt,tot,cetlit,xbt,ybt,zbt,xmt,ymt,zmt,xet,yet,zet,azbt,dipbt,jt):
                out.write('%s,' %idt[jt])
                out.write('%s,' %from_depth)
                out.write('%s,' %to_depth)
                out.write('%s,' %cetlit[jt])
                out.write('%s,' %xbt[jt])
                out.write('%s,' %ybt[jt])
                out.write('%s,' %zbt[jt])

def print_xyz_Begin_Prev_csv(): #out,idt,fromt,tot,cetlit,xbt,ybt,zbt,xmt,ymt,zmt,xet,yet,zet,azbt,dipbt,jt):
                out.write('%s,' %idt[jt])
                out.write('%s,' %from_depth)
                out.write('%s,' %to_depth)
                out.write('%s,' %cetlit[jt])
                out.write('%s,' %prev_dict["x"])
                out.write('%s,' %prev_dict["y"])
                out.write('%s,' %prev_dict["z"])

def print_xyz_Mid_csv():
                out.write('%s,' %xmt[jt])
                out.write('%s,' %ymt[jt])
                out.write('%s,' %zmt[jt])
                

def print_xyz_End_csv():
                out.write('%s,' %xet[jt])
                out.write('%s,' %yet[jt])
                out.write('%s,' %zet[jt])
                out.write('%s,' %azbt[jt])
                out.write('%s,' %dipbt[jt])
                #out.write('%s,' %azmt[jt])
                #out.write('%s,' %dipmt[jt])
                #out.write('%s,' %azet[jt])
                #out.write('%s,' %dipet[jt])
                out.write('\n')
    #out.close()

def print_xyz_mid_inEnd():  # if mid is arbitrary in intermediate value
    out.write('%s,' %xet[jt])
    out.write('%s,' %yet[jt])
    out.write('%s,' %zet[jt])



    

# Function to find distance 
def distance(x1, y1, z1, x2, y2, z2):
    d = math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) +
                math.pow(z2 - z1, 2)* 1.0) 
    return(d) 
  


#to verify the x,y,z of lithology with lepfrog data.
def Diff_XYZ():
    Calculated_Data= pd.read_csv('DB_Lithology_Export_Calc.csv',encoding = "ISO-8859-1", dtype='object')
    Leapfrog_Data= pd.read_csv('DB_Lithology_Export_Leapfrog.csv',encoding = "ISO-8859-1", dtype='object')
    Calculated_Data.xbt=Calculated_Data['xbt'].astype(float)
    Calculated_Data.ybt=Calculated_Data['ybt'].astype(float)
    Calculated_Data.zbt=Calculated_Data['zbt'].astype(float)
    Leapfrog_Data.start_x=Leapfrog_Data['start_x'].astype(float)
    Leapfrog_Data.start_y=Leapfrog_Data['start_y'].astype(float)
    Leapfrog_Data.start_z=Leapfrog_Data['start_z'].astype(float)

    Calculated_Data.xmt=Calculated_Data['xmt'].astype(float)
    Calculated_Data.ymt=Calculated_Data['ymt'].astype(float)
    Calculated_Data.zmt=Calculated_Data['zmt'].astype(float)
    Leapfrog_Data.mid_x=Leapfrog_Data['mid_x'].astype(float)
    Leapfrog_Data.mid_y=Leapfrog_Data['mid_y'].astype(float)
    Leapfrog_Data.mid_z=Leapfrog_Data['mid_z'].astype(float)

    Calculated_Data.xet=Calculated_Data['xet'].astype(float)
    Calculated_Data.yet=Calculated_Data['yet'].astype(float)
    Calculated_Data.zet=Calculated_Data['zet'].astype(float)
    Leapfrog_Data.end_x=Leapfrog_Data['end_x'].astype(float)
    Leapfrog_Data.end_y=Leapfrog_Data['end_y'].astype(float)
    Leapfrog_Data.end_z=Leapfrog_Data['end_z'].astype(float)

    ######################

    x1=Calculated_Data['xbt'].values
    y1=Calculated_Data['ybt'].values
    z1=Calculated_Data['zbt'].values
    x2=Leapfrog_Data['start_x'].values
    y2=Leapfrog_Data['start_y'].values
    z2=Leapfrog_Data['start_z'].values

    x3=Calculated_Data['xmt'].values
    y3=Calculated_Data['ymt'].values
    z3=Calculated_Data['zmt'].values
    x4=Leapfrog_Data['mid_x'].values
    y4=Leapfrog_Data['mid_y'].values
    z4=Leapfrog_Data['mid_z'].values

    x5=Calculated_Data['xet'].values
    y5=Calculated_Data['yet'].values
    z5=Calculated_Data['zet'].values
    x6=Leapfrog_Data['end_x'].values
    y6=Leapfrog_Data['end_y'].values
    z6=Leapfrog_Data['end_z'].values


    x1_count= x1.shape[0]
    y1_count= y1.shape[0]
    z1_count= z1.shape[0]
    x2_count= x2.shape[0]
    y2_count= y2.shape[0]
    z2_count= z2.shape[0]


    x3_count= x3.shape[0]
    y3_count= y3.shape[0]
    z3_count= z3.shape[0]
    x4_count= x4.shape[0]
    y4_count= y4.shape[0]
    z4_count= z4.shape[0]


    x5_count= x5.shape[0]
    y5_count= y5.shape[0]
    z5_count= z5.shape[0]
    x6_count= x6.shape[0]
    y6_count= y6.shape[0]
    z6_count= z6.shape[0]

    Diff_1=[]
    Diff_2=[]
    Diff_3=[]

    if(x1_count == y1_count==z1_count==x2_count==y2_count==z2_count):
        for c1 in range(x1_count):
            p1 = distance(x1[c1],y1[c1],z1[c1],x2[c1],y2[c1],z2[c1])
            #print(p1)
            Diff_1.append(p1)
    #print(Diff_1)
    #print('########')


    if(x3_count == y3_count==z3_count==x4_count==y4_count==z4_count):
        for c2 in range(x3_count):
            p2 = distance(x3[c2],y3[c2],z3[c2],x4[c2],y4[c2],z4[c2])
            Diff_2.append(p2)
    #print(Diff_2)
    #print('########')


    if(x5_count == y5_count==z5_count==x6_count==y6_count==z6_count):
        for c3 in range(x5_count):
            p3 = distance(x5[c3],y5[c3],z5[c3],x6[c3],y6[c3],z6[c3])
            Diff_3.append(p3)
    #print(Diff_3)
    #print('########')

    Calculated_Data['Diff_1']=Diff_1
    #print(Diff_1)
    Calculated_Data['Diff_2']=Diff_2
    Calculated_Data['Diff_3']=Diff_3
    del Calculated_Data['Unnamed: 15']
    Calculated_Data.to_csv ('Litho_xyz_Diff.csv', index = False, header=True)








def Comments_Dic(minlong,maxlong,minlat,maxlat):
    '''
    Function selects the distinct attribute column and attribute value which matches in thesaurus 'thesaurus_geology_comment' with the given region
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - List with extracted data matching attribute column and thesaurus.
    '''
    'distict on(attributecol,attributeval) changes to only attributevalue'
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
        #print(record)
        #Var.Comments_dic.append(record)
        Var.Comments_dic.append(record)     #append to Comments_dic_tmp , since we need to take another variable ,use in split fun.
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query, bounds)
   
    #with open('Dic_Comments.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
    cur.close()
    conn.close()

def Comments_Dic_Process(minlong,maxlong,minlat,maxlat):
    '''
    Function selects the distinct attribute column and attribute value which matches in thesaurus 'thesaurus_geology_comment' with the given region
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - List with extracted data matching attribute column and thesaurus.
    '''
    'distict on(attributecol,attributeval) changes to only attributevalue'
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
        #print(record)
        #Var.Comments_dic.append(record)
        Var.Comments_dic_tmp.append(record)     #append to Comments_dic_tmp , since we need to take another variable ,use in split fun.
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query, bounds)
   
    #with open('Dic_Comments.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
    cur.close()
    conn.close()


def listoflist_comments_dic():
    '''
        Function gets the listoflist for Comments_dic_tmp , so that it can be used in split function.
        Input - Comments_dic_tmp.
        output - Comments_dic.
    '''
    Var.Comments_dic = [list(elem) for elem in Var.Comments_dic_tmp]
    
 




def Comments_dic_litho_split(dic_litho_comments,filename,Comm_final_split_process_list,Num_worker_process):
    '''
    Function split listoflist to  the number of logical process considered 
    Input : 
         - dic_litho_comments : input  which needs to be split.
         -filename : Each split is printed to a file for verification.
         - Process_list : list to hold the split variables name for later use.
         - Num_worker_process : No worker process decided to select.
    Output:
        - Comments Dictionary splits in globals variables and in csv file.
    '''
    
    
    length_list= len(dic_litho_comments)
    partition_List = length_list / Num_worker_process   #split total data by process selected to make eaual chunks.
    actual_part_num = round(partition_List)    # split value with avilable logical process
    print(length_list)
    count=0
    if length_list > 0:      
        x=0
        y=length_list
        for i in range(x,y,actual_part_num):       
            x=i
            count = count + 1
            if count  == Num_worker_process  :  ## to merge last split with previous one as it is small
                total_split_val =(round(partition_List) * Num_worker_process)
                diff = length_list - total_split_val

                if diff > 0 or diff == 0 :
                    final_split = x+actual_part_num+diff
                    #print(final_split)
                    globals()[filename+ '_' + str(i)] = dic_litho_comments[x:final_split] #create global variable for later use
                    Comm_final_split_process_list.append(globals()[filename+ '_' + str(i)])  # add to process list 
                    #print("in final -1")
                    break           # exit after last split , since we added left out records

                elif diff < 0 :
                    final_split = x+ actual_part_num+diff 
                    globals()[filename+ '_' + str(i)] = dic_litho_comments[x:final_split]
                    Comm_final_split_process_list.append(globals()[filename+ '_' + str(i)])
                    #print("in final-2")
                    break
               
            else:
                globals()[filename+ '_' + str(i)] = dic_litho_comments[x:x+actual_part_num]
                Comm_final_split_process_list.append(globals()[filename+ '_' + str(i)])
                #print(" Not in final")
            

    #print(count) 

    # create csv file for verification. uncomment if require files.
    #part_num = actual_part_num
    #partnum1= part_num
    #tot_partnum = part_num
    #for x in range(0, count, 1):  # create csv file for verification. uncomment if require files.
        #if x > 0 :
                
            #var_name1 = filename+ '_' + str(tot_partnum)
            #print(var_name1)
            #my_df1 = pd.DataFrame(globals()[var_name1])  
            #file_name1 = var_name1 + '.csv'
            #my_df1.to_csv(os.path.join(export_path ,file_name1), index=False, header=True)  # create csv file for verification
            #tot_partnum = tot_partnum + part_num
                
        #elif x == 0:
            #var_name2 = filename + '_' + str(x)
            #my_df2 = pd.DataFrame(globals()[var_name2])   
            #file_name2 = var_name2 + '.csv'
            #my_df2.to_csv(os.path.join(export_path ,file_name2), index=False, header=True) # create csv file for verification
   


def Comments_With_fuzzy_Process(q,comment_split, Litho_dico,file_name): 
    '''
    Function to find the fuzzywuzzy and score for each of the split with comments attribute value.This is the function which is called by Process funtion.
    Input : 
        q - To fill the fuzzywuzzy results from each process.
        comments_split - comments split to get fuzzywuzzy.
        Litho_Dico - pass Litho_Dico to get fuzzywuzzy as Process dont share memory.
        file_name- print each fuzzywuzzy to a csv file for varification.
    Output:
        - List and csv file with fuzzywuzzy and score for comments attribute value.
    '''
    
    #print(" B P")
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
            #prinnt(" B P")
        else:
            Comments_fuzzy_Sub.append([Comments_Dic_ele[0],Comments_Dic_ele[1],cleaned_text,'Other',bestmatch])  #top_new[1]])
            #print("B P")
            


    
    #print to csv file for verification
    #my_df2 = pd.DataFrame(Comments_fuzzy_Sub , columns = ['Comments_Field','Comment_Attr_val','Comment_cleaned_text','Comment_Fuzzy_wuzzy','Comment_Score'])
    #my_df2.to_csv(os.path.join(export_path ,file_name), index=False, header=True)
    q.put(Comments_fuzzy_Sub)
    #time.sleep(1)   


def Comments_With_fuzzy():
    '''
    Function find the fuzzywuzzy and score to the comments attribute value 
    Input : 
        List with attribute column and attribute value.
    Output:
        - List with fuzzywuzzy and score for comments attribute value.
    '''
    
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    comments_sub_list=[]
    fieldnames=['Comments_Field','Comment_Attr_val','Comment_cleaned_text','Comment_Fuzzy_wuzzy','Comment_Score']
    out= open(os.path.join(export_path,"Comments_fuzzy.csv"), "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Comments_Dic_new = [list(elem) for elem in Var.Comments_dic]
    for Comments_Dic_ele in Comments_Dic_new:
        cleaned_text=clean_text(Comments_Dic_ele[1])
        
        words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
        words=words.rstrip('\n\r').split(" ")
        last=len(words)-1 #position of last word in phrase
        
        for litho_dico_ele in Var.Litho_dico:
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
            Var.Comments_fuzzy.append([Comments_Dic_ele[0],Comments_Dic_ele[1],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
            out.write('%s,' %Comments_Dic_ele[0].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))
            out.write('%s,' %Comments_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%s,' %bestlitho.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('%d,' %bestmatch)
            out.write('\n')
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
        else:
            Var.Comments_fuzzy.append([Comments_Dic_ele[0],Comments_Dic_ele[1],cleaned_text,'Other',bestmatch])  #top_new[1]])
            out.write('%s,' %Comments_Dic_ele[0].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))
            out.write('%s,' %Comments_Dic_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','').replace(',' , '').replace('\n',''))
            out.write('%s,' %cleaned_text)   #.replace('(','').replace(')','').replace('\'','').replace(',','').replace('\n',''))
            out.write('Other,')
            out.write('%d,' %bestmatch)
            out.write('\n')
            top.clear()
            CET_Litho=''
            bestmatch=-1
            bestlitho=''
                   
                    
   

def Final_Lithology_With_Comments(DB_lithology_With_Comments_Final_Export,minlong,maxlong,minlat,maxlat):
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
    cur = conn.cursor()
    #Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query)  #,Bounds)
    
    #print(cur)
    First_Filter_list = [list(elem) for elem in cur]
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Company_Lithocode','Company_Lithology','CET_Lithology','Score', 'Comment', 'CET_Comment', 'Comment_Score']
    out= open(os.path.join(export_path,DB_lithology_With_Comments_Final_Export), "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
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
        
        
        for Attr_val_fuzzy_ele in Var.Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                Company_Lithocode=Attr_val_fuzzy_ele[1]
                Company_Lithology=Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Lithology=Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Score=Attr_val_fuzzy_ele[5]
                
        for Comments_fuzzy_ele in Var.Comments_fuzzy:
            if Comments_fuzzy_ele[1] == First_filter_ele[7]:
                Comment=Comments_fuzzy_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Comment=Comments_fuzzy_ele[3].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Comment_Score=Comments_fuzzy_ele[4]
                
        #if not(Score==0 and Comment_Score==0):
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






def Final_Lithology_With_Comments_Split():  #pass the longitude and lattitude directly in the query as its join of two query.
    '''
    Function Extracts data from tables dhgeologyattr,dhgeology,collar,clbody and attribute column lithology table from DB for the specified region.
    Also joins extraction of Comments attribute column with Comments attribute value.The extracted data is split using split funtion to create processes.
    
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - split list of dataset in Final_split_proc_list.
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
    Comments_dic_litho_split(First_Filter_list,filename_final,Var.Final_split_proc_list,worker_proc)
    
    cur.close()
    conn.close()



def Final_comments_with_fuzzy_Process(split_List,Comments_fuzzy,Attr_val_fuzzy,q2,filename):
    '''
        For Each row extracted for a region, the from and to depth values are validated , generated fuzzywuzzy values for the lithology along with the score are printed.
        This is the function which is called by Process funtion.
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

    ## create csv file for verification
    df_final_split = pd.DataFrame(final_fuzzy_list)  #, index=var_name1.keys())
    df_final_split.to_csv(os.path.join(export_path ,filename), index=False, header=True) 
    
    q2.put(final_fuzzy_list)




def Final_lithology_Only_Comments(DB_lithology_Only_Comments_Final_Export,minlong,maxlong,minlat,maxlat):
    '''
    Function Extracts data from tables dhgeologyattr,dhgeology,collar,clbody and comments attribute column lithology table from DB for the specified region.
    For Each row extracted the from and to depth values are validated , generated fuzzywuzzy values for the comments lithology along with the score are printed .
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - csv file with the extracted data with fuzzywuzzy and score.
    '''
    query = '''  select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
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
                 WHERE(t3.longitude BETWEEN %s AND %s) AND(t3.latitude BETWEEN %s AND %s) 
                 ORDER BY t3.companyid ASC '''
                 
                 
                 
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query,Bounds)
        
    First_Filter_list = [list(elem) for elem in cur]
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comment', 'CET_Comment', 'Comment_Score']
    out= open(os.path.join(export_path,DB_lithology_Only_Comments_Final_Export), "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    for First_filter_ele in First_Filter_list:
        if (First_filter_ele[0] == None and First_filter_ele[1]== None  and First_filter_ele[2]== None  and First_filter_ele[3]== None) or  (First_filter_ele[2] == None and First_filter_ele[3] ==None) :   # for empty fields
            continue
            
        else:    
            First_filter_ele[2],First_filter_ele[3] =Depth_validation_comments(First_filter_ele[2],First_filter_ele[3])  #,First_filter_ele[0],First_filter_ele[2],logger1) # validate depth
            CompanyID=First_filter_ele[0]
            CollarID=First_filter_ele[1]
            FromDepth=First_filter_ele[2]
            ToDepth=First_filter_ele[3]
            Comment=""
            CET_Comment=""
            Comment_Score=0
        
        
       
                
        for Comments_fuzzy_ele in Var.Comments_fuzzy:
            if Comments_fuzzy_ele[1] == First_filter_ele[6]:
                Comment=Comments_fuzzy_ele[1].replace('(','').replace(')','').replace('\'','').replace(',','')
                CET_Comment=Comments_fuzzy_ele[3].replace('(','').replace(')','').replace('\'','').replace(',','')  #.replace(',' , ''))
                Comment_Score=Comments_fuzzy_ele[4]
                
        #if not(Score==0 and Comment_Score==0):
        out.write('%d,' %CompanyID)
        out.write('%d,' %CollarID)
        out.write('%d,' %FromDepth)
        out.write('%s,' %ToDepth)
        out.write('%s,' %Comment)
        out.write('%s,' %CET_Comment)
        out.write('%d,' %Comment_Score)
        out.write('\n')
    cur.close()
    conn.close()
    out.close()
 
    

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
    new_table_name.to_csv('../data/export_db/join.csv')















