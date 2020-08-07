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
import globalvariables as var




#Attr_col_collar_dic_list=[]



            
    
            
            
def collar_collar_attri_Final(DB_Collar_Export,src_csr,dst_csr,minlong,maxlong,minlat,maxlat):
   #print("-----start Final---")

   fieldnames=['CollarID','HoleId','Longitude','Latitude','RL','MaxDepth','X','Y']
   out= open(DB_Collar_Export, "w",encoding ="utf-8")
   for ele in fieldnames:
        out.write('%s,' %ele)
   out.write('\n')
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
    #create tranformer object
   transformer = Transformer.from_crs(src_csr, dst_csr)
    
   
  
   
   try:
      conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
      cur = conn.cursor()
      Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
      cur.execute(query,Bounds)
      collar_collarAttr_Filter = [list(elem) for elem in cur]
      DicList_collar_collarattr = [list(elem) for elem in var.Attr_col_collar_dic_list]
      for collar_ele in collar_collarAttr_Filter:
         #if (collar_ele[0] == 305574):
            #print("its danger")
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
                     #chk large , with empty case, write old rec to file
                     
                     if(len(list_rl)!=0):
                        
                        RL = maximum(list_rl,'NAN')
                     else:
                        RL = maximum(list_rl,'NAN')
                        #RL = "NAN"
                     if(len(list_maxdepth)!=0):
                        
                        Maxdepth = maximum(list_maxdepth,'NAN')
                     else:
                         Maxdepth = maximum(list_maxdepth,'NAN')
                         #Maxdepth ="NAN"
                         
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
                        #RL ="NAN"
                     if(len(list_maxdepth)!=0):
                        
                        Maxdepth = maximum(list_maxdepth,'NAN')
                     else:
                         Maxdepth = maximum(list_maxdepth,'NAN')
                         #Maxdepth = "NAN"


                     write_to_csv = True  
        
                     Cur_id =collar_ele[0]
                     Cur_hole_id = collar_ele[1]
                     Cur_Longitude =collar_ele[2]
                     Cur_latitude = collar_ele[3]

                     list_maxdepth.clear()
                     list_rl.clear()
                     
                     list_maxdepth.append(Parse_Num(collar_ele[5]))
                     
        
         
         x2,y2=transformer.transform(Pre_latitude,Pre_Longitude) # tranform long,latt
         if(write_to_csv == True):
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
            Cur_id = 0
          

         else:
            continue
          
   
      cur.close()
   except (Exception, psycopg2.DatabaseError) as error:
      print(error)
   finally:
      if conn is not None:
         conn.close()

   #print("-----End Final---")




def Parse_Num(s1):
   s1=s1.lstrip()
   if re.match("^[-+]?[0-9]+$", s1):
      return(int(s1))
   elif re.match("[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?", s1):
      return(float(s1))
   elif s1.isalpha():
      return(None)



def maximum(iterable, default):
  #   '''Like max(), but returns a default value if xs is empty.'''
    try:
        return str(max(i for i in iterable if i is not None))
    except ValueError:
        return default





def collar_attr_col_dic():
   #print("------ dictionary----start")
   query =""" SELECT  rl_maxdepth_dic.attributecolumn,rl_maxdepth_dic.cet_attributecolumn  FROM rl_maxdepth_dic """
   conn = None
   
   try:
      conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
      cur = conn.cursor()
      cur.execute(query)

      for rec in cur:
         var.Attr_col_collar_dic_list.append(rec)

   
      #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
      #with open('Dic_attr_col_collar.csv', 'w',encoding="utf-8") as f:
         #cur.copy_expert(outputquery, f)
      
 
      cur.close()
   except (Exception, psycopg2.DatabaseError) as error:
      print(error)
   finally:
      if conn is not None:
         conn.close()

   #print("------ dictionary----End")
         









import psycopg2
import csv
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import math
import datetime
import pandas as pd
from math import acos, cos, asin, sin, atan2, tan, radians

#Attr_col_survey_dic_list=[]

def Survey_Final(DB_Survey_Export,minlong,maxlong,minlat,maxlat):
   #print("-----start Final---")
   fieldnames=['CollarID','Depth','Azimuth','Dip']
   out= open(DB_Survey_Export, "w",encoding ="utf-8")
   for ele in fieldnames:
        out.write('%s,' %ele)
   out.write('\n')
   query =""" select t1.collarid,t1.depth,t2.attributecolumn,t2.attributevalue 
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
      conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
      cur = conn.cursor()
      Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
      cur.execute(query,Bounds)
      Survey_First_Filter = [list(elem) for elem in cur]
      Survey_dic_list = [list(elem) for elem in var.Attr_col_survey_dic_list] 
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


                     
               if ('AZI' in attr_col_ele[1] and (Pre_id ==0 or Pre_id ==survey_ele[0])): # and back_survey_1 == survey_ele[1] ):   #AZI
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
                           DIP=-90 #default Dip to -90
                           AZI = float((survey_ele[3]).replace('\'','').strip().rstrip('\n\r'))
                           AZI_sub_list.append(AZI)
                           back_survey_0 =survey_ele[0]
                           back_survey_1 = survey_ele[1]
                           One_AZI =False
                           
                           
                           

               if ('DIP' in attr_col_ele[1] and (Pre_id ==survey_ele[0] or Pre_id ==0)) :   #DIP
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

   #print("-----End Final---")




def Attr_col_dic():
   #print("------ dictionary----start")
   query =""" SELECT * FROM public.survey_dic """
   conn = None
   temp_list =[]
   try:
      conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
      cur = conn.cursor()
      cur.execute(query)

      for rec in cur:
         var.Attr_col_survey_dic_list.append(rec)

         
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

   #print("------ dictionary----End")
         




def count_Digit(n):
    if n > 0:
        digits = int(math.log10(n))+1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n))+1 # +1 if you don't count the '-'
  
    return digits


def convert_survey(DB_Collar_Export,DB_Survey_Export,DB_Survey_Export_Calc):
   location=pd.read_csv(DB_Collar_Export)
   survey=pd.read_csv(DB_Survey_Export)
   survey=pd.merge(survey,location, how='left', on='CollarID')

   fieldnames=['CollarID','Depth','Azimuth','Dip','X','Y','Z']
   out= open(DB_Survey_Export_Calc, "w",encoding ="utf-8")
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
         X2,Y2,Z2=dia2xyz(X1,Y1,Z1,last_Dip,last_Azi,last_Depth,float(row['Dip']),float(row['Azimuth']),float(row['Depth']))
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











import psycopg2
import csv
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import math
from collections import Counter
import datetime
import pandas as pd
import numpy as np
from math import acos, cos, asin, sin, atan2, tan, radians
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords


#First_Filter_list=[['11410',3118047,169.7,169.9,'Lithology','GR'],['11410',3118060,22,23,'Lithology','CL']]
#First_Filter_list=[]
#Attr_col_list=[]
#Litho_dico=[]
#cleanup_dic_list=[]
#Att_col_List_copy_tuple=[]
#Attr_val_Dic=[]
#Attr_val_fuzzy=[]




def Attr_Val_Dic():
    #print("------------------start Dic_Attr_val------------")
    query = """SELECT * FROM public.dic_attr_val_lithology_filter"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        var.Attr_val_Dic.append(record)
    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
    #with open('Dic_attr_val.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
    

    cur.close()
    conn.close()

    #print("------------------end Dic_Attr_val------------")


   



def Litho_Dico():
    #print("------------------Start Litho_Dico------------")
    query = """SELECT litho_dic_1.clean  FROM litho_dic_1"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    #print(cur)
    for record in cur:
        #print(record)
        var.Litho_dico.append(record)
        #print(Litho_dico)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('Dic_litho.csv', 'w') as f:
        #cur.copy_expert(outputquery, f)
        
    #print(Litho_dico)
    cur.close()
    conn.close()
    #print("------------------end Litho_Dico------------")


    
    

def Clean_Up():
    #print("------------------start Clean_Up_Dico------------")


    query = """SELECT cleanup_lithology.clean  FROM cleanup_lithology"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        var.cleanup_dic_list.append(record)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('cleanup_dic.csv', 'w',encoding="utf-8") as f:
        #cur.copy_expert(outputquery, f)
        

    cur.close()
    conn.close()

    #print("------------------End Clean_Up_Dico------------")

  




def clean_text(text):
    text=text.lower().replace('unnamed','').replace('meta','').replace('meta-','').replace('undifferentiated ','').replace('unclassified ','')
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
    text = text.replace('\'', '')
    text = text.replace('\\', '')                        
	
    for cleanup_dic_ele in var.cleanup_dic_list:
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
    #print("--------start of Attr_val_fuzzy-----------")
    bestmatch=-1
    bestlitho=''
    top=[]
    i=0
    attr_val_sub_list=[]
    fieldnames=['CollarID','code','Attr_val','cleaned_text','Fuzzy_wuzzy','Score']
    out= open("Attr_val_fuzzy.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Attr_val_Dic_new = [list(elem) for elem in var.Attr_val_Dic]
    for Attr_val_Dic_ele in Attr_val_Dic_new:
        
        cleaned_text_1=clean_text(Attr_val_Dic_ele[2])
        cleaned_text_1=tokenize_and_lemma(cleaned_text_1)
        cleaned_text=" ".join(str(x) for x in cleaned_text_1)

        #cleaned_text=clean_text(Attr_val_Dic_ele[2])   # for testing purpose
        
        #if(cleaned_text =='granite'):
            #print(cleaned_text)
        words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
        words=words.rstrip('\n\r').split(" ")
        last=len(words)-1 #position of last word in phrase
        for Litho_dico_ele in var.Litho_dico:
            #print(Litho_dico)
            litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').split(" ")
            #print(litho_words)
            #if(litho_words == "alkali-feldspar-granite"):
                #print("Alkali-feldspar-granite")


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
                        #top=top+sc
                        top.append([sc[0],sc[1]])
        
               
            
                
        
           
            
            
        if bestmatch >80:
            
            var.Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,bestlitho,bestmatch]) #top_new[1]])  or top[0][1]
                             
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
            var.Attr_val_fuzzy.append([Attr_val_Dic_ele[0],Attr_val_Dic_ele[1],Attr_val_Dic_ele[2],cleaned_text,'Other',bestmatch])  #top_new[1]])
            
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
            
     #print("--------End of Attr_val_fuzzy-----------")



def Depth_validation(row_2,row_3):
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
    #print("--------start of Final -----------")
    query = """select t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn, t1.attributevalue 
		 from public.dhgeologyattr t1 
		 inner join public.dhgeology t2 
		 on t1.dhgeologyid = t2.id 
		 inner join collar t3 
		 on t3.id = t2.collarid 
		 inner join clbody t4 
		 on t4.companyid = t3.companyid
		 inner join public.dic_att_col_lithology_1 t5
		 on t1.attributecolumn = t5.att_col
		 WHERE(t3.longitude BETWEEN %s AND %s) AND(t3.latitude BETWEEN %s AND %s) 
		 ORDER BY t3.companyid ASC"""


    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query,Bounds)
    var.First_Filter_list = [list(elem) for elem in cur]
    #print("First Filter ready")
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']
    out= open(DB_Lithology_Export, "w",encoding ="utf-8")
    #out_first_filter= open("DB_lithology_First.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    #Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for First_filter_ele in var.First_Filter_list:
        for Attr_val_fuzzy_ele in var.Attr_val_fuzzy:
            if int(Attr_val_fuzzy_ele[0].replace('\'' , '')) == First_filter_ele[0] and  Attr_val_fuzzy_ele[1].replace('\'' , '') == First_filter_ele[5]:
                #print(Attr_val_fuzzy_ele[0],"\t",Attr_val_fuzzy_ele[1])
                #print(First_filter_ele[0],"\t",First_filter_ele[5])
                First_filter_ele[2],First_filter_ele[3] =Depth_validation(First_filter_ele[2],First_filter_ele[3])
                out.write('%d,' %First_filter_ele[0])
                out.write('%d,' %First_filter_ele[1])
                out.write('%d,' %First_filter_ele[2])
                out.write('%s,' %First_filter_ele[3])
                out.write('%s,' %Attr_val_fuzzy_ele[1])
                out.write('%s,' %Attr_val_fuzzy_ele[2].replace('(','').replace(')','').replace('\'','').replace(',',''))
                out.write('%s,' %Attr_val_fuzzy_ele[4].replace('(','').replace(')','').replace('\'','').replace(',',''))   #.replace(',' , ''))
                out.write('%d,' %int(Attr_val_fuzzy_ele[5]))
                out.write('\n')

    
        #for column in First_filter_ele:
            #out_first_filter.write('%s,' %column)
        #out_first_filter.write('\n')
        	
	
   # print("--------End of Final -----------")


def Upscale_lithology(DB_Lithology_Export,DB_Lithology_Upscaled_Export):
    #print("--------start of Upsacle -----------")
    Hierarchy_litho_dico_List =[]
    query = """ select * from public.hierarchy_dico """
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    Hierarchy_litho_dico_List  = [list(elem) for elem in cur]
    CET_hierarchy_dico = pd.DataFrame(Hierarchy_litho_dico_List,columns=['Level_3','Level_2','Level_1'])
    DB_Lithology= pd.read_csv(DB_Lithology_Export,encoding = "ISO-8859-1", dtype='object')
    Upscaled_Litho=pd.merge(DB_Lithology, CET_hierarchy_dico, left_on='CET_Lithology', right_on='Level_3')
    Upscaled_Litho.sort_values("Company_ID", ascending = True, inplace = True)
    #Upscaled_Litho.drop(['Unnamed: 8'], axis=1)
    del Upscaled_Litho['Unnamed: 8']
    Upscaled_Litho.to_csv (DB_Lithology_Upscaled_Export, index = False, header=True)
    
    #print("--------End of Upsacle -----------")



def Remove_duplicates_Litho(DB_Lithology_Upscaled_Export,Upscaled_Litho_NoDuplicates_Export):
    Final_Data= pd.read_csv(DB_Lithology_Upscaled_Export)   
    Final_Data.CollarID = Final_Data.CollarID.astype(int)
    Final_Data.Fromdepth = Final_Data.Fromdepth.astype(float)
    Final_Data.Todepth = Final_Data.Todepth.astype(float)
    Final_Data.sort_values(['CollarID', 'Fromdepth','Todepth'], inplace=True)
    singles = Final_Data.drop_duplicates(subset=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode'],keep='first',inplace =False)   #,'Company_Lithology','CET_Lithology','Score'
    singles.to_csv(Upscaled_Litho_NoDuplicates_Export,index=False)











