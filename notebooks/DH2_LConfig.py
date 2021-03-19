from datetime import datetime
#from striplog import Legend, Lexicon, Interval, Component, Striplog
import os
import multiprocessing as mp
#from itertools import count

host_ = '130.95.198.59'
port_= 5432
DB_='gswa_dh'
user_ = 'postgres'
pwd_ = 'loopie123pgpw'
#export_path = ''


#Extents to query , bounds for lithology+comments needs to be provided in query as it has two query with join.
#ranee
minlong=115.5
maxlong=118
minlat=-30.5
maxlat=-27.5

#fabilea
#minlong=121.2
#maxlong=122.89
#minlat=-21.04
#maxlat=-21.03


#src_pro,Dst_proj
src_csr = 4326 
dst_csr = 28350


#Number of worker process
Tot_workers = mp.cpu_count()
worker_proc = Tot_workers - 2    #Tot_workers is Operating system provided process ,user can modify this to use as many process as required.




#ExportFiles
#one_time = True
#if one_time == True:
 #   nowtime=datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%p")   #.isoformat(timespec='minutes')
  #  export_path ='../data/export_db/'+ nowtime
   # one_time = False
    
#print(export_path)
#def only_once():
    #nowtime=datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%p")   #.isoformat(timespec='minutes')
    #export_path ='../data/export_db/'+ nowtime
   # if not os.path.exists(export_path): 
     #   os.mkdir(export_path)
        
    #only_once.func_code = (lambda:None).func_code
    

nowtime=datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%p")   #.isoformat(timespec='minutes')
export_path ='../data/export_db/'+ nowtime
if not os.path.exists(export_path): 
    os.mkdir(export_path)

#nowtime=datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%p")   #.isoformat(timespec='minutes')
#export_path ='../data/export_db/'+ nowtime
#os.mkdir(export_path)


export_path_1 = '../data/export_db/'


#csv , log files with time stamp
DB_Collar_Rl_Log = 'DB_Collar_Rl_Log' +   '.log'
DB_Collar_Maxdepth_Log =  'DB_Collar_maxdepth_Log' + '.log'
DB_Collar_Export= 'DB_Collar_Export' +  '.csv'   #.replace("-","/").replace("T"," At ")
DB_Survey_Azi_Log =  'DB_Survey_Azi_Log' +  '.log'
DB_Survey_Dip_Log =  'DB_Survey_Dip_Log' +  '.log'
DB_Survey_Export= 'DB_Survey_Export'+ '.csv'
DB_Survey_Export_Calc= 'DB_Survey_Export_Calc'+ '.csv'
CET_Litho= 'CET_Litho'+ '.csv'
DB_Litho_Depth_Log =  'DB_Litho_Depth_Log' +  '.log'
DB_Litho_Att_Val_Log =  'DB_Litho_Att_Val_Log' +  '.log'
DB_Lithology_Export= 'DB_Lithology_Export'+ '.csv'
DB_lithology_With_Comments_Final_Export =  'DB_lithology_With_Comments_Final' +  '.csv'
DB_lithology_Only_Comments_Final_Export =  'DB_lithology_Only_Comments_Final' +  '.csv'
DB_Lithology_Export_Backup= 'DB_Lithology_Export_Backup'+ '.csv'
DB_Lithology_Upscaled_Export= 'DB_Lithology_Upscaled_Export'+ '.csv'
Upscaled_Litho_NoDuplicates_Export = 'Upscaled_Litho_NoDuplicates_Export'+ '.csv'
DB_Lithology_Export_Calc= 'DB_Lithology_Export_Calc'+ '.csv'
DB_Lithology_Export_VTK= 'DB_Lithology_Export'+ '.vtp'

#DB_Lithology_Export_Calc=export_path_1+  'DB_Lithology_Export_Calc.csv'
#DB_Lithology_Export_VTK= export_path_1+ 'DB_Lithology_Export.vtp'





#csv , log files without time stamps
#DB_Collar_Rl_Log = export_path + 'DB_Collar_Rl_Log.log'
#DB_Collar_Maxdepth_Log = export_path + 'DB_Collar_maxdepth_Log.log'
#DB_Collar_Export=export_path + '//'+ 'DB_Collar_Export.csv'   #.replace("-","/").replace("T"," At ")
#DB_Survey_Export=export_path+'DB_Survey_Export.csv'
#DB_Survey_Export_Calc=export_path+'DB_Survey_Export_Calc.csv'
#CET_Litho=export_path+'CET_Litho.csv'
#DB_Lithology_Export=export_path+'DB_Lithology_Export.csv'
#DB_lithology_With_Comments_Final_Export = export_path + 'DB_lithology_With_Comments_Final.csv'
#DB_lithology_Only_Comments_Final_Export = export_path + 'DB_lithology_Only_Comments_Final.csv'
#DB_Lithology_Export_Backup=export_path+'DB_Lithology_Export_Backup.csv'
#DB_Lithology_Upscaled_Export=export_path+'DB_Lithology_Upscaled_Export.csv'
#Upscaled_Litho_NoDuplicates_Export = export_path+'Upscaled_Litho_NoDuplicates_Export.csv'
#DB_Lithology_Export_Calc=export_path+'DB_Lithology_Export_Calc.csv'
#DB_Lithology_Export_VTK=export_path+'DB_Lithology_Export.vtp'

#DB_Lithology_Export_Calc=export_path+'DB_Lithology_Export_Calc.csv'
#DB_Lithology_Export_VTK=export_path+'DB_Lithology_Export.vtp'







print('Default parameters loaded from DH2_LConfig.py:')
with open('../notebooks/DH2_LConfig.py', 'r') as myfile:
  data = myfile.read()
  print(data)
  myfile.close()
print('\nModify these parameters in the cell below')


