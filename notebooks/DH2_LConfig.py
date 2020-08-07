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
export_path='../data/export_db/'
DB_Collar_Export=export_path+'DB_Collar_Export.csv'
DB_Survey_Export=export_path+'DB_Survey_Export.csv'
DB_Survey_Export_Calc=export_path+'DB_Survey_Export_Calc.csv'
CET_Litho=export_path+'CET_Litho.csv'
DB_Lithology_Export=export_path+'DB_Lithology_Export.csv'
DB_Lithology_Export_Backup=export_path+'DB_Lithology_Export_Backup.csv'
DB_Lithology_Upscaled_Export=export_path+'DB_Lithology_Upscaled.csv'
Upscaled_Litho_NoDuplicates_Export=export_path +'Upscaled_Litho_NoDuplicates.csv'
DB_Lithology_Export_Calc=export_path+'DB_Lithology_Export_Calc.csv'
DB_Lithology_Export_VTK=export_path+'DB_Lithology_Export.vtp'

print('Default parameters loaded from DH2_LConfig.py:')
with open('DH2_LConfig.py', 'r') as myfile:
  data = myfile.read()
  print(data)
  myfile.close()
print('\nModify these parameters in the cell below')


