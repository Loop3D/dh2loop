from ipyleaflet import Map, basemaps, GeoJSON, LayersControl, DrawControl,WMSLayer
from ipywidgets import Label
import ipywidgets as widgets
import geopandas as gpd
import folium
import pandas as pd
import csv
import requests
import json
import random
from shapely.geometry import Polygon
from datetime import datetime
import os
from urllib.error import HTTPError
import psycopg2

def draw_interactive_map():
    """
    Draws interactive map to be able to draw/define a region of interest
    """
    wms_drillholes = WMSLayer(
        url='http://geo.loop-gis.org/geoserver/loop/wms?',
        layers='loop:collar_4326',
        format='image/png',
        transparent=True,
        attribution='Drilhole collar from GSWA',
        name='drillhole collars'
    )
    
    wms_geol = WMSLayer(
        url='http://geo.loop-gis.org/geoserver/loop/wms?',
        layers='loop:2_5m_interpgeop15_4326',
        format='image/png',
        transparent=True,
        opacity=0.4,
        attribution='Geology data from GSWA',
        name='geology'
    )
    m =Map(basemap=basemaps.OpenTopoMap, center=(-29,116.5), zoom=8,scroll_wheel_zoom=True)

    m.add_layer(wms_geol)
    m.add_layer(wms_drillholes)

    m.add_control(LayersControl())
    dc = DrawControl(rectangle={'shapeOptions': {'color': '#0000FF'}})
    m.add_control(dc)
    m
    
def define_bounds(bounds):
    """
    Extracts bounds from region drawn/defined
	Args:
		`bounds`= json bounds captured
    Returns:
        `bbox`= bounds of the region defined (list)
		`bbox2`= bounds of the region defined (string)
    """
    
    #ew_poly=GeoJSON(data=dc.last_draw)
    new_poly=str(bounds)

    if("'geometry': None" in new_poly):
        raise NameError('Error: No rectangle selected')
    new_poly=new_poly.rsplit("'coordinates': ", 1)[1]
    new_poly=new_poly.replace('[[[','').replace('[','').replace(']]]}})','').replace('],','').replace(',','').split(" ")
    longs=new_poly[0::2]
    lats=new_poly[1::2]
    minlong=float(min(longs))
    maxlong=float(max(longs))
    minlat=float(max(lats)) #ignores sign
    maxlat=float(min(lats)) #ignores sign
    bbox2 = str(minlong)+","+str(minlat)+","+str(maxlong)+","+str(maxlat)
    bbox =(minlong,minlat,maxlong,maxlat)
    bounds2=(minlong,maxlong,minlat,maxlat)
    #bbox =[minlong,minlat,maxlong,maxlat]
    #print(bbox2)
    print("Bounds:", bbox)
    return bbox, bbox2, bounds2
    
def query_anumbers(bbox,bbox2,bounds2):
    """
    Queries anumbers of the reports within region defined
    Args:
        `bbox`= bounds of the region defined
    Returns:
        `anumberscode`=list of anumbers
    """
    try:
        collars_file='http://geo.loop-gis.org/geoserver/loop/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=loop:collar_4326&bbox='+bbox2+'&srs=EPSG:4326'
        collars = gpd.read_file(collars_file, bbox=bbox)
        print("Connected to Loop Server")
        anumbers=gpd.GeoDataFrame(collars, columns=["anumber"])
        anumbers = pd.DataFrame(anumbers.drop_duplicates(subset=["anumber"]))
    except HTTPError as err:
        if err.code == 404 or err.code == 500 or err.code == 503:
            query="""SELECT DISTINCT (collar.anumber)
			FROM public.collar
			WHERE(longitude BETWEEN %s AND %s) AND
			(latitude BETWEEN %s AND %s)
			ORDER BY collar.anumber ASC"""

            conn = psycopg2.connect(host="130.95.198.59", port = 5432, 
			database="gswa_dh", user="postgres", password="loopie123pgpw")
            cur = conn.cursor()
            cur.execute(query, bounds2)
            anumbers=pd.DataFrame(cur, columns=["anumber"])
            print("Connected to PostgreSQL Server")
        else:
            raise
    #collars_file='http://geo.loop-gis.org/geoserver/loop/wfs?service=WFS&version=1.0.0&request=GetFeature&typeName=loop:collar_4326&bbox='+bbox2+'&srs=EPSG:4326'
    #collars = gpd.read_file(collars_file, bbox=bbox)
    #anumbers=gpd.GeoDataFrame(collars, columns=["anumber"])
    #anumbers = pd.DataFrame(anumbers.drop_duplicates(subset=["anumber"]))
    #print(anumbers)
    anumbers['anumberlength']=anumbers['anumber'].astype(str).map(len)
    anumberscode=[]
    for index, row in anumbers.iterrows():
        if (int(row[1])==5):
            text=str("a0"+ str(row[0]))
            text2=str("a"+ str(row[0]))
        elif (int(row[1])==4):
            text=str("a00"+ str(row[0]))
            text2=str("a"+ str(row[0]))
        elif (int(row[1])==3):
            text=str("a000"+ str(row[0]))
            text2=str("a"+ str(row[0]))
        elif (int(row[1])==2):
            text=str("a0000"+ str(row[0]))
            text2=str("a"+ str(row[0]))
        elif (int(row[1])==1):
            text=str("a00000"+ str(row[0]))
            text2=str("a"+ str(row[0]))
        else:
            text= str("a"+ str(row[0]))
        anumberscode.append(text)
        anumberscode.append(text2)
    print("Report Numbers:", anumberscode)
    return anumberscode

def get_links(anumberscode):
    """
    Finds CloudStor links for selected anumbers
    Args:
        `anumberscode`=list of anumbers
    Returns:
        `FilteredList`= list of corresponding links
    """
    try:
        FileDirectory=pd.read_csv('https://geo.loop-gis.org/files/FileDirectory.csv',encoding = "ISO-8859-1", dtype='object')
        print("Connected to Loop Server")		
    except HTTPError as err:
        if err.code == 404 or err.code == 500 or err.code == 503:
            FileDirectory=pd.read_csv('https://docs.google.com/spreadsheets/d/1NJwknGclMQq96N9igVCi67D0cjdoGlggqkpeldtrQ9o/edit?usp=sharing',encoding = "ISO-8859-1", dtype='object')
            print("Connected to Google Drive Backup")
        else:
            raise
	#FileDirectory=pd.read_csv('https://geo.loop-gis.org/files/FileDirectory.csv',encoding = "ISO-8859-1", dtype='object')
    FilteredList = FileDirectory[FileDirectory.Files.str.contains('|'.join(anumberscode))]
    FilteredList = FilteredList ['Directory']
    return FilteredList
    #print(FilteredList)
    #https://gist.github.com/wragge/d6250f0c61196ebe76121cfdc4bdafe2

def download_reports(FilteredList):
    """
    Downloads reports from CloudStor links
    Args:
        `FilteredList`= list of corresponding links
    """
    nowtime=datetime.now().isoformat(timespec='minutes')
    dir=os.getcwd()
    if(not os.path.isdir('../data/')):
        os.mkdir('../data/')	
    directory='../data/downloaded_reports'+'_'+nowtime.replace("-","").replace(":","").replace("T","_")+'/'
    os.mkdir(directory)
    for url in FilteredList.iteritems():
        url = url[1]
        r = requests.get(url, allow_redirects=True)
        if url.find('%2F'):
            filename=url.rsplit('%2F', 1)[1]
        open(directory + filename, 'wb').write(r.content)
    return directory
	
def get_reports(bounds):
    """
    Downloads reports from a defined region
    """
    bbox, bbox2, bounds2 =define_bounds(bounds)
    anumberscode=query_anumbers(bbox, bbox2, bounds2)
    if not anumberscode:
        print ("No reports available in this region")
    else:
        FilteredList=get_links(anumberscode)
        directory=download_reports(FilteredList)
        print("Download Complete. Find files at: "+ directory)
    
