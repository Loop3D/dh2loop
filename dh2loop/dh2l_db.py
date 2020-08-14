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
import datetime
import pandas as pd
import numpy as np
from math import acos, cos, asin, sin, atan2, tan, radians


Attr_col_collar_dic_list=[]



            
    
            
            
def collar_collar_attri_Final(DB_Collar_Export,src_csr,dst_csr,minlong,maxlong,minlat,maxlat):
  # print("-----start Final---")

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
      DicList_collar_collarattr = [list(elem) for elem in Attr_col_collar_dic_list]
      for collar_ele in collar_collarAttr_Filter:
         #if (collar_ele[0] == 305574):
            #print("its danger")
         for Dic_ele in DicList_collar_collarattr:
            if(collar_ele[4] == Dic_ele[0]):
               
               if(Dic_ele[1] == 'rl'):
                  #print("1")
                  if(Pre_id== collar_ele[0] or Pre_id ==0 or Cur_id ==collar_ele[0]):
                     #print("2")
                     list_rl.append(Parse_Num(collar_ele[5]))
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
 
                     Cur_id =collar_ele[0]
                     Cur_hole_id = collar_ele[1]
                     Cur_Longitude =collar_ele[2]
                     Cur_latitude = collar_ele[3]

                     list_rl.clear()
                     list_maxdepth.clear()
                     
                     list_rl.append(Parse_Num(collar_ele[5]))
                     
             
               elif(Dic_ele[1]=='maxdepth'):
                  #print("7")
                  if(Pre_id== collar_ele[0] or Pre_id == 0 or Cur_id ==collar_ele[0] ):
                     if(collar_ele[5][0] == '-'):
                        #print("7")
                        list_maxdepth.append(Parse_Num(collar_ele[5])*-1)
                     else:
                        #print("8")
                        list_maxdepth.append(Parse_Num(collar_ele[5]))

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
         Attr_col_collar_dic_list.append(rec)

   
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

Attr_col_survey_dic_list=[]

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
         Attr_col_survey_dic_list.append(rec)

         
      #Attr_col_survey_dic_list = [list(elem) for elem in temp_list]

      #for ele in Attr_col_survey_dic_list:
         #print(ele)
         #Attr_col_survey_dic_list.append(record)

      outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
      with open('Dic_attr_col_survey.csv', 'w') as f:
         cur.copy_expert(outputquery, f)
      
 
          
 
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
First_Filter_list=[]
Attr_col_list=[]
Litho_dico=[]
cleanup_dic_list=[]
Att_col_List_copy_tuple=[]
Attr_val_Dic=[]
Attr_val_fuzzy=[]


#print("------------------start Dic_Attr_Col------------")
def Attr_COl():
    query = """SELECT * FROM public.dic_att_col_lithology_1"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
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

    #print("------------------end Dic_Attr_Col------------")




#print("------------------start Dic_Attr_val------------")
def Attr_Val_Dic():
    query = """SELECT * FROM public.dic_attr_val_lithology_filter"""
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    for record in cur:
        #print(record)
        Attr_val_Dic.append(record)
    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
   
    with open('Dic_attr_val.csv', 'w') as f:
        cur.copy_expert(outputquery, f)
    

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
        Litho_dico.append(record)
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
        cleanup_dic_list.append(record)
    #outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query)
       
    #with open('cleanup_dic.csv', 'w',encoding="utf-8") as f:
        #cur.copy_expert(outputquery, f)
        

    cur.close()
    conn.close()

    #print("------------------End Clean_Up_Dico------------")

  



def First_Filter():
    print("------------------start First_Filter------------")
    start = time.time()
    #out= open("DB_lithology_First1.csv", "w",encoding ="utf-8")
    query = """select t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn, t1.attributevalue 
    from public.dhgeologyattr t1 
    inner join public.dhgeology t2 
    on t1.dhgeologyid = t2.id 
    inner join collar t3 
    on t3.id = t2.collarid 
    inner join clbody t4 
    on t4.companyid = t3.companyid 
    WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN - 30.5 AND - 27.5) 
    ORDER BY t3.companyid ASC"""


    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
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
    out.close() 
    end = time.time()
    print(end - start)
    print("------------------End First_Filter------------")




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
	
    for cleanup_dic_ele in cleanup_dic_list:
        cleaned_item =str(cleanup_dic_ele).replace('(','').replace(')','').replace(',','').replace('\'','')
        text = text.replace('cleaned_item','')
    return text








#Final File
def Final_Lithology_old():
    print("--------start of Final -----------")
    bestmatch=-1
    bestlitho=''
    top=[]
    p = re.compile(r'[- _]')
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']
    out= open("DB_lithology_Final.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    query = '''SELECT dic_attr_val_lithology_filter.company_id,dic_attr_val_lithology_filter.company_code,replace(dic_attr_val_lithology_filter.comapany_litho, ',' , '_') as comapany_litho  FROM dic_attr_val_lithology_filter'''
    conn = psycopg2.connect(host='130.95.198.59', port = 5432, database='gswa_dh', user='postgres', password='loopie123pgpw')
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
                cleaned_text=clean_text(row[2])
                #print(cleaned_text)
                words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
                words=words.split(" ")
                last=len(words)-1 #position of last word in phrase
                
                for Litho_dico_ele in Litho_dico:              
                    #litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').split(" ")
                    litho_words=re.split(p, str(Litho_dico_ele))
                    scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
                    for sc in scores:                        
                        if(sc[1]>bestmatch): #better than previous best match
                            bestmatch =  sc[1]
                            bestlitho=litho_words[0]
                            top=sc
                            if(sc[0]==words[last]): #bonus for being last word in phrase
                                bestmatch=bestmatch*1.01
                        elif (sc[1]==bestmatch): #equal to previous best match
                            if(sc[0]==words[last]): #bonus for being last word in phrase
                                bestlitho=litho_words[0]
                                bestmatch=bestmatch*1.01
                            else:
                                top=top+sc

                #top = [list(elem) for elem in top]
                top_new = list(top)
                if top_new[1] >80:
                    #del First_filter_ele[4]
                    #del First_filter_ele[4]
                    #for column in First_filter_ele:
                    out.write('%s,' %First_filter_ele[0])
                    out.write('%s,' %First_filter_ele[1])
                    out.write('%s,' %(First_filter_ele[2]).replace(',' ,' '))
                    out.write('%s,' %First_filter_ele[3])
                    out.write('%s,' %row[1])
                    out.write('%s,' %row[2])
                    CET_Litho = str(top_new[0]).replace('(','').replace(')','').replace('\'','').replace(',','')
                    CET_Litho = CET_Litho.replace(',', ' ')
                    out.write('%s,' %CET_Litho)
                    out.write('%d,' %top_new[1])
                    out.write('\n')
                    #top.clear()
                    top_new[:] =[]
                    CET_Litho=''
                    bestmatch=-1
                    bestlitho=''
                else:
                    #del First_filter_ele[4]
                    #del First_filter_ele[4]
                    #for column in First_filter_ele:
                    out.write('%s,' %First_filter_ele[0])
                    out.write('%s,' %First_filter_ele[1])
                    out.write('%s,' %(First_filter_ele[2]).replace(',' ,' '))
                    out.write('%s,' %First_filter_ele[3])
                    out.write('%s,' %row[1])
                    out.write('%s,' %row[2])
                    out.write('Other,')
                    out.write('%d,' %top_new[1])
                    out.write('\n')
                    #top.clear()
                    top_new[:] =[]
                    CET_Litho=''
                    bestmatch=-1
                    bestlitho=''

    cur.close()
    conn.close()
    out.close()
    print("--------End of Final-----------")



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
    #p = re.compile(r'[' ']')
    fieldnames=['CollarID','code','Attr_val','cleaned_text','Fuzzy_wuzzy','Score']
    out= open("Attr_val_fuzzy.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for Attr_val_Dic_ele in Attr_val_Dic_new:
        

       # Attr_val_Dic_ele[2] = Attr_val_Dic_ele[2].replace('\'','').replace('-','').replace('/','')
       # cleaned_text_1=tokenize_and_lemma(Attr_val_Dic_ele[2])  #tokenize
       # print(cleaned_text_1)
       # cleaned_text_2=" ".join(str(x) for x in cleaned_text_1)  #convert to string from list
       # print(cleaned_text_2)
       # cleaned_text=clean_text(cleaned_text_2)   #tokenize

        cleaned_text_1=clean_text(Attr_val_Dic_ele[2])
        cleaned_text_1=tokenize_and_lemma(cleaned_text_1)
        cleaned_text=" ".join(str(x) for x in cleaned_text_1)

        #cleaned_text=clean_text(Attr_val_Dic_ele[2])   # for testing purpose
        
        #if(cleaned_text =='granite'):
            #print(cleaned_text)
        words=(re.sub('\(.*\)', '', cleaned_text)).strip() 
        words=words.rstrip('\n\r').split(" ")
        last=len(words)-1 #position of last word in phrase
        for Litho_dico_ele in Litho_dico:
            #print(Litho_dico)
        #litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').split(" ")
            #litho_words=re.split(" ", str(Litho_dico_ele))
            #litho_words=str(Litho_dico_ele).split(" ")
            litho_words=str(Litho_dico_ele).lower().rstrip('\n\r').replace('(','').replace(')','').replace('\'','').replace(',','').split(" ")
            #print(litho_words)
            #if(litho_words == "alkali-feldspar-granite"):
                #print("Alkali-feldspar-granite")


            scores=process.extract(cleaned_text, litho_words, scorer=fuzz.token_set_ratio)
            for sc in scores:                        
                if(sc[1]>bestmatch): #better than previous best match
                    bestmatch =  sc[1]
                    bestlitho=litho_words[0]
                    #print(bestmatch)
                    #print(bestlitho)
                    #top=sc
                    top.append([sc[0],sc[1]])
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestmatch=bestmatch*1.01
                        #print("inside 1")
                        #print(sc[0])
                        #print(words[last])
                elif (sc[1]==bestmatch): #equal to previous best match
                    if(sc[0]==words[last]): #bonus for being last word in phrase
                        bestlitho=litho_words[0]
                        bestmatch=bestmatch*1.01
                        #print(bestlitho)
                        #print(bestmatch)
                        #print(words[last])
                    else:
                        #top=top+sc
                        top.append([sc[0],sc[1]])
        
        #print(top)
        #top_new = list(top)
        #top_new=[list(elem) for elem in top]
        #for i in range(len(top)):
            
        #print(top_new)
        i=0
        #print(" %s %d " %(top_new[0], top_new[1] ))

        
        #for top_new_ele in top:
            #if(top_new_ele[0].replace('(','').replace(')','').replace('\'','').replace(',','') == cleaned_text):
               # bestlitho = cleaned_text
               # bestmatch = 100
                
            
                
        
           
            
            
        if bestmatch >80:
            #CET_Litho = str(top_new[0]).replace('(','').replace(')','').replace('\'','').replace(',','')
            #print(CET_Litho)
            
            #attr_val_sub_list.append(Attr_val_Dic_ele[0])
            #attr_val_sub_list.append(Attr_val_Dic_ele[1])
            #attr_val_sub_list.append(Attr_val_Dic_ele[2])
            #attr_val_sub_list.append(bestlitho)
            #attr_val_sub_list.append(top_new[1])
            #Attr_val_fuzzy.append(attr_val_sub_list)

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
            #attr_val_sub_list.append(Attr_val_Dic_ele[0])
            #attr_val_sub_list.append(Attr_val_Dic_ele[1])
            #attr_val_sub_list.append(Attr_val_Dic_ele[2])
            #attr_val_sub_list.append('Other')
            #attr_val_sub_list.append(top_new[1])
            #Attr_val_fuzzy.append(attr_val_sub_list)
            #attr_val_sub_list.clear()


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
   # print("--------start of Final -----------")
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
		 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN - 30.5 AND - 27.5) 
		 ORDER BY t3.companyid ASC"""


    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    First_Filter_list = [list(elem) for elem in cur]
    print("First Filter ready")
    fieldnames=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode','Company_Lithology','CET_Lithology','Score']
    out= open(DB_Lithology_Export, "w",encoding ="utf-8")
    #out_first_filter= open("DB_lithology_First.csv", "w",encoding ="utf-8")
    for ele in fieldnames:
        out.write('%s,' %ele)
    out.write('\n')
    #Attr_val_Dic_new = [list(elem) for elem in Attr_val_Dic]
    for First_filter_ele in First_Filter_list:
        for Attr_val_fuzzy_ele in Attr_val_fuzzy:
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
        	
	
    #print("--------End of Final -----------")


def Upscale_lithology(DB_Lithology_Export,DB_Lithology_Upscaled_Export):
    #print("--------start of Upsacle -----------")
    Hierarchy_litho_dico_List =[]
    query = """ select * from public.hierarchy_dico """
    conn = psycopg2.connect(host="130.95.198.59", port = 5432, database="gswa_dh", user="postgres", password="loopie123pgpw")
    cur = conn.cursor()
    cur.execute(query)
    Hierarchy_litho_dico_List  = [list(elem) for elem in cur]
    CET_hierarchy_dico = pd.DataFrame(Hierarchy_litho_dico_List,columns=['Level_3','Level_2','Level_1'])
    #CET_hierarchy_dico.to_csv ('CET_hierarchy_dico.csv', index = False, header=True)
    #print (CET_hierarchy_dico)
    DB_Lithology= pd.read_csv(DB_Lithology_Export,encoding = "ISO-8859-1", dtype='object')
    Upscaled_Litho=pd.merge(DB_Lithology, CET_hierarchy_dico, left_on='CET_Lithology', right_on='Level_3')
    Upscaled_Litho.sort_values("Company_ID", ascending = True, inplace = True)
    #Upscaled_Litho.drop(['Unnamed: 8'], axis=1)
    del Upscaled_Litho['Unnamed: 8']
    Upscaled_Litho.to_csv (DB_Lithology_Upscaled_Export, index = False, header=True)
    
    #Upscaled_Litho= Upscaled_Litho.loc[:, ~Upscaled_Litho.columns.str.contains('^Unnamed')]
    #Upscaled_Litho.reset_index(level=0, inplace=True)
    #Upscaled_Litho['CET_Litho']=Upscaled_Litho['index']
    #del Upscaled_Litho['index']
    #Upscaled_Litho.to_csv(DB_Lithology_Upscaled)
    #print("--------End of Upsacle -----------")



def Remove_duplicates_Litho(DB_Lithology_Upscaled_Export,Upscaled_Litho_NoDuplicates_Export):
    Final_Data= pd.read_csv(DB_Lithology_Upscaled_Export)   
    Final_Data.CollarID = Final_Data.CollarID.astype(int)
    Final_Data.Fromdepth = Final_Data.Fromdepth.astype(float)
    Final_Data.Todepth = Final_Data.Todepth.astype(float)
    Final_Data.sort_values(['CollarID', 'Fromdepth','Todepth'], inplace=True)
    singles = Final_Data.drop_duplicates(subset=['Company_ID','CollarID','Fromdepth','Todepth','Comapny_Lithocode'],keep='first',inplace =False)   #,'Company_Lithology','CET_Lithology','Score'
    singles.to_csv(Upscaled_Litho_NoDuplicates_Export,index=False)







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

def convert_lithology():
    print("--------start of convert Lithology -----------")
    collar= pd.read_csv('DB_Collar_Export.csv',encoding = "ISO-8859-1", dtype='object')
    survey= pd.read_csv('DB_Survey_Export.csv',encoding = "ISO-8859-1", dtype='object')
    litho= pd.read_csv('Upscaled_Litho.csv',encoding = "ISO-8859-1", dtype='object')
    
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
    out= open('DB_Lithology_Export_Calc.csv', "w",encoding ="utf-8")
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
                if(idc[jc] == '7777'):
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
        

        prev_dict ={"todepth" : -0.0,
                            "x": -0.0,
                            "y": -0.0,
                            "z": -0.0}

        

        for jt in range(indt, nt):
            #if(idc[jc]!=idt[jt]):
                #break
            if idc[jc]==idt[jt]:
                #print(idt[jt])
                global temp_litho_df
                #temp_litho_df = (litho.loc[litho['CollarID'].isin([idt[jt]])]).copy()
                if litho_sub_cnt_flag == True :  # to keep track of count till the end of one collarID
                    temp_litho_df = (litho.loc[litho['CollarID']== idt[jt]])
                    #filtered_df = df.loc[df['Symbol'] == 'A99']
                    litho_sub_cnt=temp_litho_df.shape[0]
                    litho_sub_cnt_flag = False

                    
                    strt_pos = temp_Survey_df[temp_Survey_df['Depth']== 0].index.item()
                    #print(start_pos)
                    #print(temp_Survey_df)
                    

                else :
                    tmp_litho_sub_cnt = tmp_litho_sub_cnt + 1
                    
                #if(litho_sub_cnt > survey_sub_cnt)
                    
                #print(litho_sub_cnt)
                #print(temp_litho_df)
                #print(tot)
                #print(idc[jc])

                if(idc[jc] == '7777'):
                    ii =ii + 1

                
                indt = jt


                
                Depth_survey = temp_Survey_df['Depth'].values
                litho_Todepth = temp_litho_df['Todepth'].values

                from_depth = fromt[jt]
                mid_depth = float(fromt[jt]) + float((float(tot[jt])-float(fromt[jt]))/2)
                to_depth = tot[jt]

                

                begin_f = True
                mid_f = True
                end_f  = True

                

                print(idt[jt])
               
                #print(tot[jt] in temp_Survey_df.Depth.values)

                if(tot[jt] in temp_Survey_df.Depth.values  and tot[jt] > Depth_survey [survey_ind+1]  ):
                    #tmp_litho_sub_cnt = litho_sub_cnt -1
                   #loc = np.where(Depth_survey [survey_ind] ==tot[jt])
                    print(tot[jt])
                    end_pos = temp_Survey_df[temp_Survey_df['Depth']==tot[jt]].index.item()
                    #from_depth = fromt[jt]
                    #to_depth = tot[jt]
                    #fromt[jt] = Depth_survey [survey_ind]
                    #print(pos)
                    print(tot[jt])
                    while (start_pos < end_pos):
                        print(survey_ind)
                        #print(Depth_survey [survey_ind])
                        fromt[jt] = Depth_survey [survey_ind]
                        print(fromt[jt])
                        tot[jt] = Depth_survey [survey_ind + 1]
                        print(tot[jt])
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
                    print("Max of Survey", "\t",max(Depth_survey))
                    #if survey_sub_cnt-1 == survey_ind+1 and tmp_litho_sub_cnt == litho_sub_cnt :
                        #litho_more_ele = True
                        
                    #if  survey_ind+1 == survey_sub_cnt-1 and tmp_litho_sub_cnt == litho_sub_cnt :
                        #litho_more_ele = True
                        
                    if survey_sub_cnt-1 == survey_ind+1 and tmp_litho_sub_cnt < litho_sub_cnt :
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
                           print(survey_ind)
                           #print(survey_sub_cnt-2)
                           #print(last_Litho_ToDepth)
                           #tot[jt] = last_Litho_ToDepth

                           #if tmp_litho_sub_cnt == litho_sub_cnt :
                               #fromt[jt] = from_depth
                               #tot[jt] = to_depth
                               #calculate_x_y_z()
                               #break



                           
                           fromt[jt] = Depth_survey [survey_ind]
                           print(fromt[jt])
                           tot[jt] = to_depth
                           print(tot[jt])
                           calculate_x_y_z()
                           #survey_ind = survey_ind +1
                           break
                           #print_xyz_csv()
                           #survey_ind = survey_ind +1
                           
                       else:
                           print(survey_ind)
                           fromt[jt] = Depth_survey [survey_ind]
                           print(fromt[jt])
                           tot[jt] = Depth_survey [survey_ind + 1]
                           print(tot[jt])
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

                elif(tot[jt] > Depth_survey [survey_ind+1]):
                    while True:
                        if Depth_survey [survey_ind] <=  tot[jt]  >= Depth_survey [survey_ind+1]:
                           survey_ind = survey_ind +1
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
                print(mid)
                azm2, dip2 = angleson1dh(indbs,indes,ats,azs,dips,mid)
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
                if from_depth == fromt[jt] and begin_f == True :
                    print_xyz_Begin_csv()
                    begin_f = False

                print(prev_dict)
                print(begin_f)
                    
                if prev_dict["todepth"] == from_depth and begin_f == True and  survey_ind == survey_sub_cnt-2  :
                    print(prev_dict)
                    print(jt)
                    print(survey_sub_cnt-1)
                    print(survey_ind)
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

                

                if prev_dict["todepth"] == from_depth and mid_f == True and  survey_ind == survey_sub_cnt-2  :
                    print_xyz_Mid_csv()
                    mid_f =False
                    
                xet[jt] = float(x)+float(xet[jt])
                yet[jt] = float(y)+float(yet[jt])
                zet[jt] = float(z)+float(zet[jt])
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
                print("survey Index","\t",survey_ind)

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


    





                   
                    
   

    
    
















