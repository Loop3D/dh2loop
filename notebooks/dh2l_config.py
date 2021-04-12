

host_ = '130.95.198.59'
port_= 5432
DB_='gswa_dh'
user_ = 'postgres'
pwd_ = 'loopie123pgpw'
#export_path = ''


#File encoding type
encoding_1 ="utf-8"
encoding_2 = "ISO-8859-1"

#Database Files
data_path='../data/'
wamex_path='../data/wamex/'
collar_file=wamex_path+'collar.csv'
collarattr_file=wamex_path+'collarattr.csv'
dhsurvey_file=wamex_path+'dhsurvey.csv'
dhsurveyattr_file=wamex_path+'dhsurveyattr.csv'
dhgeology_file=wamex_path+'dhgeology.csv'
dhgeologyattr_file=wamex_path+'dhgeologyattr.csv'

#Thesauri
rl_maxdepth_dic_file= wamex_path+'rl_maxdepth_dic.csv'
survey_dic_file=wamex_path+'survey_dic.csv'
dic_attr_col_lithology_file=wamex_path+'dic_att_col_lithology.csv'
dic_attr_val_lithology_file=wamex_path+'dic_attr_val_lithology_filter.csv'
cleanup_lithology_file=wamex_path+'cleanup_lithology.csv'
litho_dic_file=wamex_path+'litho_dic_1.csv'
CET_hierarchy_dico_file=wamex_path+'hierarchy_dico.csv'

#ExportFiles
export_path='../data/export/'
DB_Collar_Export=export_path+'DB_Collar_Export.csv'
DB_Survey_Export=export_path+'DB_Survey_Export.csv'
DB_Survey_Export_Calc=export_path+'DB_Survey_Export_Calc.csv'
CET_Litho=export_path+'CET_Litho.csv'
DB_Lithology_Export=export_path+'DB_Lithology_Export.csv'
DB_Lithology_Export_Backup=export_path+'DB_Lithology_Export_Backup.csv'
DB_Lithology_Upscaled=export_path+'DB_Lithology_Upscaled.csv'
DB_Lithology_Export_Calc=export_path+'DB_Lithology_Export_Calc.csv'
DB_Lithology_Export_VTK=export_path+'DB_Lithology_Export.vtp'


#Shapefiles
shapefile_path='../data/shapefile/'
geology=shapefile_path+'500K_interpgeol16_yalgoo_singleton.shp'

#Projections
src_crs = {'init': 'EPSG:4326'}  # coordinate reference system for imported dtms (geodetic lat/long WGS84)
dst_crs = {'init': 'EPSG:28350'} # coordinate system for data

print('Default parameters loaded from dh2l_config.py:')
with open('../notebooks/dh2l_config.py', 'r') as myfile:
  data = myfile.read()
  print(data)
  myfile.close()
print('\nModify these parameters in the cell below')