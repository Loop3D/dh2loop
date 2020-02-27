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
dic_attr_val_lithology_filter_file=wamex_path+'dic_attr_val_lithology_filter.csv'
litho_dic_1_file=wamex_path+'litho_dic_1,csv'
cleanup_lithology_file=wamex_path+'cleanup_lithology.csv'
dic_att_col_lithology_file=wamex_path+'dic_att_col_lithology.csv'
CET_hierarchy_dico_file=wamex_path+'hierarchy_dico.csv'

#ExportFiles
export_path='../data/export/'
DB_Collar_Export=export_path+'DB_Collar_Export.csv'
DB_Survey_Export=export_path+'DB_Survey_Export.csv'
CET_Litho=export_path+'CET_Litho.csv'
DB_Lithology_Export=export_path+'DB_Lithology_Export.csv'
DB_Lithology_Export_Backup=export_path+'DB_Lithology_Export_Backup.csv'
DH_Lithology_Upscaled=export_path+'DH_Lithology_Upscaled.csv'

#Shapefiles
shapefile_path='../data/shapefile/'
geology=shapefile_path+'500K_interpgeol16_yalgoo_singleton.shp'

print('Default parameters loaded from dh2l_config.py:')
with open('../notebooks/dh2l_config.py', 'r') as myfile:
  data = myfile.read()
  print(data)
  myfile.close()
print('\nModify these parameters in the cell below')