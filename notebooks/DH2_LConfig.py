from datetime import datetime
#Extents to query 
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


#ExportFiles
nowtime=datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%p")   #.isoformat(timespec='minutes')
export_path='../data/export_db/'
DB_Collar_Rl_Log = export_path + 'DB_Collar_Rl_Log_' + nowtime + '.log'
DB_Collar_Maxdepth_Log = export_path + 'DB_Collar_maxdepth_Log_' + nowtime +'.log'
DB_Collar_Export=export_path+'DB_Collar_Export_' + nowtime + '.csv'   #.replace("-","/").replace("T"," At ")
DB_Survey_Export=export_path+'DB_Survey_Export_'+ nowtime +'.csv'
DB_Survey_Export_Calc=export_path+'DB_Survey_Export_Calc_'+ nowtime +'.csv'
CET_Litho=export_path+'CET_Litho_'+ nowtime +'.csv'
DB_Lithology_Export=export_path+'DB_Lithology_Export_'+ nowtime +'.csv'
DB_lithology_With_Comments_Final_Export = export_path + 'DB_lithology_With_Comments_Final' + nowtime + '.csv'
DB_lithology_Only_Comments_Final_Export = export_path + 'DB_lithology_Only_Comments_Final' + nowtime + '.csv'
DB_Lithology_Export_Backup=export_path+'DB_Lithology_Export_Backup_'+ nowtime +'.csv'
DB_Lithology_Upscaled_Export=export_path+'DB_Lithology_Upscaled_Export_'+ nowtime +'.csv'
Upscaled_Litho_NoDuplicates_Export = export_path+'Upscaled_Litho_NoDuplicates_Export_'+ nowtime +'.csv'
DB_Lithology_Export_Calc=export_path+'DB_Lithology_Export_Calc_'+ nowtime +'.csv'
DB_Lithology_Export_VTK=export_path+'DB_Lithology_Export_'+ nowtime +'.vtp'

print('Default parameters loaded from DH2_LConfig.py:')
with open('../notebooks/DH2_LConfig.py', 'r') as myfile:
  data = myfile.read()
  print(data)
  myfile.close()
print('\nModify these parameters in the cell below')


