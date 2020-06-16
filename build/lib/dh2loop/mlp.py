####ALL CODES FROM GEOVEC_LITHO
#### EDITED TO FIT ANY DATASET /OUR DATASET
####https://github.com/IFuentesSR/GeoVectoLitho
#from __future__ import division

import nltk
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
from glove import Glove
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

import pyproj
import geopandas as gpd
import os

import json
import h5py

from keras.models import load_model
from scipy import interpolate
from itertools import product
import gdal
import ogr

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

LabelEncoder = LabelEncoder()
one_enc = OneHotEncoder()

##Notes
#OWN=Company_Litho;Comment
#reclass=CET_Litho;CET_Comment
#x=xmt
#y=ymt
#z=zmt
#topdepth=xbt
#bottomdepth=zet

def litho_Dataframe(csv, output_pkl):
    '''Function that creates a single georeferenced dataframe with
    lithologic descriptions
    Input:
        -csv: csv to groundwater explorer files
    Output:
        -output_pkl: pandas dataframe with georeferenced lithologic descriptions'''

    DF1 = pd.read_csv(csv)
    DF2 = DF1[['Company_Litho','CET_Litho','Comment','CET_Comment','FromDepth','ToDepth',
                     'zbt', 'zet', 'xmt', 'ymt', 'zmt',]]
    DF2=DF2[DF2.duplicated('CET_Litho',keep=False)]
    DF2=DF2[DF2.duplicated('Company_Litho',keep=False)]
    DF2 = DF2.dropna(how='any')
    DF = DF2.copy()
    DF.to_pickle('{}.pkl'.format(output_pkl))
    print('number of original litho classes:', len(DF.Company_Litho.unique()))
    print('number of CET_Litho classes :', len(DF['CET_Litho'].unique()))
    print('unclassified in CET_Litho:', len(DF[DF['CET_Litho'].isnull()]))
    print('number of original comment classes:', len(DF.Comment.unique()))
    print('number of CET_Comment classes :',len(DF['CET_Comment'].unique()))
    print('unclassified in CET_Comment:',len(DF[DF['CET_Comment'].isnull()]))
    return DF

def load_geovec():
    dir=os.getcwd()
    if(not os.path.isdir('../data/')):
        os.mkdir('../data/')
    if(not os.path.isdir('../data/glove/')):
        os.mkdir('../data/glove/')
    if(not os.path.isfile('../data/glove/geovec_300d_v1.h5')):
        r = requests.get('https://www.dropbox.com/s/rsd29bqspmkopt5/geovec_300d_v1.h5?dl=0', allow_redirects=True)
        open('../data/glove/geovec_300d_v1.h5', 'wb').write(r.content)
    model = Glove()
    with h5py.File('../data/glove/geovec_300d_v1.h5', 'r') as f:
        v = np.zeros(f['vectors'].shape, f['vectors'].dtype)
        f['vectors'].read_direct(v)
        dct = f['dct'][()].tostring().decode('utf-8')
        dct = json.loads(dct)
    model.word_vectors = v
    model.no_components = v.shape[1]
    model.word_biases = np.zeros(v.shape[0])
    model.add_dictionary(dct)
    return model

def tokenize(text, min_len=1):
    '''Function that tokenize a set of strings
    Input:
        -text: set of strings
        -min_len: tokens length
    Output:
        -list containing set of tokens'''
	# Stopwords
    extra_stopwords = [
        'also',
    ]
    stop = stopwords.words('english') + extra_stopwords
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
    lemma = nltk.WordNetLemmatizer()
    lemmas = [lemma.lemmatize(t) for t in filtered_tokens]
    return lemmas


def get_vector(word, model, return_zero=False):
    '''Function that retrieves word embeddings (vector)
    Inputs:
        -word: token (string)
        -model: trained MLP model
        -return_zero: boolean variable
    Outputs:
        -wv: numpy array (vector)'''
    epsilon = 1.e-10

    unk_idx = model.dictionary['unk']
    idx = model.dictionary.get(word, unk_idx)
    wv = model.word_vectors[idx].copy()

    if return_zero and word not in model.dictionary:
        n_comp = model.word_vectors.shape[1]
        wv = np.zeros(n_comp) + epsilon

    return wv

def mean_embeddings(output_pkl,own, reclass, model):
    '''Function to retrieve sentence embeddings from dataframe with
    lithological descriptions.
    Inputs:
        -output_pkl: pandas dataframe containing lithological descriptions
                         and reclassified lithologies
        -model: word embeddings model generated using GloVe
    Outputs:
        -DF: pandas dataframe including sentence embeddings'''
    DF = pd.read_pickle(output_pkl)
    DF = DF.drop_duplicates(subset=['xmt', 'ymt', 'zmt'])

    #DF['tokens'] = DF[own].apply(lambda x: tokenize_and_lemma(x))   
    DF['tokens'] = own.apply(lambda x: tokenize_and_lemma(x))
    #DF['length'] = DF['tokens'].apply(lambda x: len(x))
    DF['length'] = DF['tokens'].str.len()
    DF = DF.loc[DF['length']> 0]
    DF['vectors'] = DF['tokens'].apply(lambda x: np.asarray([get_vector(n, model) for n in x]))
    DF['mean'] = DF['vectors'].apply(lambda x: np.mean(x[~np.all(x == 1.e-10, axis=1)], axis=0))
    #DF[reclass] = pd.Categorical(DF.reclass)
    #DF['code'] = DF.reclass.cat.codes
    #reclass = pd.Categorical(reclass)
    s=reclass.astype('category')
    #t=s.cat.codes
    #print(t)
    DF['code'] = s.cat.codes  
    DF['drop'] = DF['mean'].apply(lambda x: (~np.isnan(x).any()))
    DF = DF[DF['drop']]
    return DF

def split_stratified_dataset(DF, test_size, validation_size):
    '''Function that split dataset into test, training and validation subsets
    Inputs:
        -DF: pandas dataframe with sentence mean_embeddings
        -test_size: decimal number to generate the test subset
        -validation_size: decimal number to generate the validation subset
    Outputs:
        -X: numpy array with embeddings
        -Y: numpy array with lithological classes
        -X_test: numpy array with embeddings for test subset
        -Y_test: numpy array with lithological classes for test subset
        -Xt: numpy array with embeddings for training subset
        -yt: numpy array with lithological classes for training subset
        -Xv: numpy array with embeddings for validation subset
        -yv: numpy array with lithological classes for validation subset
        '''
    X = np.vstack(DF['mean'].values)
    Y = DF.code.values.reshape(len(DF.code), 1)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=test_size,
                                                        stratify=Y,
                                                        random_state=42)
    Xt, Xv, yt, yv = train_test_split(X_train,
                                      y_train,
                                      test_size=validation_size,
                                      stratify=None,
                                      random_state=1)
    return X, Y, X_test, y_test, Xt, yt, Xv, yv

def retrieve_predictions(classifier, x):
    '''Function that retrieves lithological classes using the trained classifier
    Inputs:
        -classifier: trained MLP classifier
        -x: numpy array containing embbedings
    Outputs:
        -codes_pred: numpy array containing lithological classes predicted'''
    preds = classifier.predict(x, verbose=0)
    u=np.unique(x)
    print(x.shape)
    print(np.size(x.shape[1]))
    new_onehot = np.zeros((x.shape[0], 20))
    new_onehot[np.arange(len(preds)), preds.argmax(axis=1)] = 1
    codes_pred = one_enc.inverse_transform(new_onehot)
    return codes_pred


def classifier_assess(classifier, x, y):
    '''Function that prints the performance of the classifier
    Inputs:
        -classifier: trained MLP classifier
        -x: numpy array with embeddings
        -y: numpy array with lithological classes predicted'''
    Y2 = retrieve_predictions(classifier, x)
    print('f1 score: ', metrics.f1_score(y, Y2, average='macro'),
          'accuracy: ', metrics.accuracy_score(y, Y2),
          'balanced_accuracy:', metrics.balanced_accuracy_score(y, Y2))


def save_predictions(DF, classifier, x, output_pred_pkl):
    '''Function that saves dataframe predictions as a pickle file
    Inputs:
        -DF: pandas dataframe with mean_embeddings
        -classifier: trained MLP model,
        -x: numpy array with embeddings,
        -output_pred_pkl: string name to save dataframe
    Outputs:
        -save dataframe'''
    preds = classifier.predict(x, verbose=0)
    DF['predicted_probabilities'] = preds.tolist()
    DF['pred'] = retrieve_predictions(classifier, x).astype(np.int32)
    DF[['xmt', 'ymt', 'FromDepth', 'ToDepth', 'zbt', 'zet',
               'mean', 'predicted_probabilities', 'pred', 'reclass', 'code']].to_pickle('{}.pkl'.format(output_pred_pkl))

def geoSurvey(Dataframe, cols_in, cols_out, Imax=0.2):
    # Imax = intervale depth
    bue = []
# loop over all the data (pick each row)
    for index, row in Dataframe.iterrows():
        if 0 < Dataframe[cols_in[2]][index]-Dataframe[cols_in[3]][index] < 1.5*Imax:
            # first condition: if thickness is very low
            a = [Dataframe[cols_in[0]][index],
                 Dataframe[cols_in[1]][index],
                 (Dataframe[cols_in[2]][index]+Dataframe[cols_in[3]][index])/2,
                 Dataframe[cols_in[4]][index],
                 Dataframe[cols_in[5]][index],
                 Dataframe[cols_in[6]][index]]
            bue.append(a)
        elif 1.5*Imax <= Dataframe[cols_in[2]][index]-Dataframe[cols_in[3]][index] <= 2.5*Imax:
            # second way, if thicknes is between two intemediate values
            a1 = [Dataframe[cols_in[0]][index],
                  Dataframe[cols_in[1]][index],
                  Dataframe[cols_in[2]][index]-Imax/2,
                  Dataframe[cols_in[4]][index],
                  Dataframe[cols_in[5]][index],
                  Dataframe[cols_in[6]][index]]

            a2 = [Dataframe[cols_in[0]][index],
                  Dataframe[cols_in[1]][index],
                  Dataframe[cols_in[3]][index]+Imax/2,
                  Dataframe[cols_in[4]][index],
                  Dataframe[cols_in[5]][index],
                  Dataframe[cols_in[6]][index]]
            bue.append(a1)
            bue.append(a2)
            # it pick two points from strata and add it to the list
        elif Dataframe[cols_in[2]][index]-Dataframe[cols_in[3]][index] > 2.5*Imax:
            # third way, if thicknes is higher than 2.5 Imax
            X = int(round(((Dataframe[cols_in[2]][index]-Imax/2)-(Dataframe[cols_in[3]][index]+Imax/2))/Imax))+1
            N = range(1, X)
            # N number of intermediate point extractions in the strata
            Ic = ((Dataframe[cols_in[2]][index]-Imax/2)-(Dataframe[cols_in[3]][index]+Imax/2))/X
            # top extraction point
            zini = [Dataframe[cols_in[0]][index],
                    Dataframe[cols_in[1]][index],
                    Dataframe[cols_in[2]][index]-Imax/2,
                    Dataframe[cols_in[4]][index],
                    Dataframe[cols_in[5]][index],
                    Dataframe[cols_in[6]][index]]

            # bottom extraction point
            zfin = [Dataframe[cols_in[0]][index],
                    Dataframe[cols_in[1]][index],
                    Dataframe[cols_in[3]][index]+Imax/2,
                    Dataframe[cols_in[4]][index],
                    Dataframe[cols_in[5]][index],
                    Dataframe[cols_in[6]][index]]

            bue.append(zini)
            bue.append(zfin)
            for n in N:
                bue.append([Dataframe[cols_in[0]][index],
                            Dataframe[cols_in[1]][index],
                            Dataframe[cols_in[2]][index]-Imax/2-(Ic*n),
                            Dataframe[cols_in[4]][index],
                            Dataframe[cols_in[5]][index],
                            Dataframe[cols_in[6]][index]])
            # loop over each of the intermediate points to extract,
            # defining x,y,z and litho in each, and finally adding to the list
        else:
            continue

    geo = pd.DataFrame(bue)
    geo.columns = cols_out
    return geo

def resampling(output_pred_pkl, output_pred_resampled_pkl):
    '''Function that creates a resampled pandas dataframe using the Gallerini &
    Donatis methodology
    Inputs:
        -output_pred_pkl: path to your MLP predicted lithological classes
    Outputs:
        -toMap: resampled pandas dataframe
        -output_pred_resampled_pkl: pkl file of resampled dataframe'''
    cols_in = ['xmt', 'ymt', 'ToDepth', 'FromDepth', 'code', 'mean', 'pred']
    cols_out = ['xmt', 'ymt', 'zmt', 'class', 'mean', 'pred']
    DF = pd.read_pickle(output_pred_pkl)
    DF['zbt'] = pd.to_numeric(DF['zbt'])
    DF['zet'] = pd.to_numeric(DF['zet'])
    DF = DF[DF.zet > -1200]
    index = range(len(DF))
    DF['ix'] = index
    DF = DF.set_index('ix')
    toMap = geoSurvey(Dataframe=DF, cols_in=cols_in, cols_out=cols_out, Imax=1)
    toMap.to_pickle('{}.pkl'.format(output_pred_resampled_pkl))
    return toMap

def find_idxs(arr, arr_B):
    ans = np.where(np.sum(np.expand_dims(arr, 0) == arr_B, 1) == 2)[0]
    if len(ans):
        return ans[0]
    return np.nan

def split_dataset(toMap):
    '''Function that split dataset into test, training and validation subsets
    Inputs:
        -toMap: resampled dataframe based on shapefile extent
    Outputs:
        -test: test subset
        -training: training subset
        -validation: validation subset'''
    grouped_DF = toMap.groupby(['xmt', 'ymt'])
    groups = np.arange(grouped_DF.ngroups)
    np.random.seed(0)
    sampling = np.random.choice(groups,
                                size=round(0.1*grouped_DF.ngroups),
                                replace=False)
    test = toMap[grouped_DF.ngroup().isin(sampling)]
    others = toMap[~grouped_DF.ngroup().isin(sampling)]
    others_grouped = others.groupby(['xmt', 'ymt'])
    others_groups = np.arange(others_grouped.ngroups)
    train_sampling = np.random.choice(others_groups,
                                      size=round(0.9*others_grouped.ngroups),
                                      replace=False)
    training = others[others_grouped.ngroup().isin(train_sampling)]
    validation = others[~others_grouped.ngroup().isin(train_sampling)]
    return test, training, validation


def get_2D(output_pred_resampled_pkl, dem_path, scale, depth_interval=1):
    '''Function that generates a numpy array with coordinates and litho classes
    Inputs:
        -output_pred_resampled_pkl: path to Gallerini resampled dataframe
        -dem_path: path to DEM raster
        -scale: scale of mapping
        -depth_interval: depth interval for mapping
    Outputs:
        -data2: numpy array with coordinates and interpolated litho classes'''

    subset = data = pd.read_pickle(output_pred_resampled_pkl)
    src = gdal.Open(dem_path)
    geo = src.GetGeoTransform()
    rb = src.GetRasterBand(1)
    test, training, validation = split_dataset(subset)
    x_min, x_max = subset.xmt.min(), subset.xmt.max()
    y_min, y_max = subset.ymt.min(), subset.ymt.max()
    z_min, z_max = subset.zmt.min(), subset.zmt.max()
    points_int_x, points_int_y = np.arange(x_min, x_max, scale), np.arange(y_min, y_max, scale)
    # y rows, x columns
    xys_toint = list(product(points_int_y, points_int_x))
    data1 = []
    for n in range(60):
        de = depth_interval
        z = np.arange(z_min+n-0.5, z_min+n+0.5, de)
        points_train = training[(training.zmt > zmt[0]*de) & (training.zmt < (zmt[0]+1)*de)]
        Ln = interpolate.LinearNDInterpolator(points_train[['ymt', 'xmt']].values,
                                              np.array(points_train['mean'].tolist()))
        Ln_int = Ln(xys_toint)
        Lns = model.predict(Ln_int, verbose=0)
        onehot_Ln = np.zeros((Lns.shape[0], 18))
        onehot_Ln[np.arange(len(Lns)), Lns.argmax(axis=1)] = 1
        codes_Ln = one_enc.inverse_transform(onehot_Ln)
        elev = []
        for m in xys_toint[:]:
            mx, my = m[1], m[0]
            px = int((mx - geo[0])/geo[1])
            py = int((my - geo[3])/geo[5])
            intval = rb.ReadAsArray(px, py, 1, 1)
            elev.append(intval[0][0])

        elev = [e - n - 0.5 for e in elev]
        yxs = np.array([[n[0], n[1]] for n in xys_toint])
        yxzs = np.hstack((yxs,
                          np.array(elev).reshape(-1, 1),
                          codes_Ln.reshape(-1, 1)))
        data1.append(yxzs)
    data2 = np.vstack(data1)
    return data2


def get_3D(geo2D, output_pred_resampled_pkl, scale, depthMask=np.nan, xMask=np.nan, yMask=np.nan):
    '''Function that generates a 3D numpy array with coordinates and litho classes
    Inputs:
        -geo2D: numpy array with coordinates and interpolated litho classes
        -scale: scale of mapping
        -output_pred_resampled_pkl:resampled dataset based on shapefile limits
        -depthMask: depth to mask 3D map
        -xMask: x coordinate to mask 3D map
        -yMask: y coordinate to mask 3D map
    Outputs:
        -recake: 3Darray to map interpolated lithologies from embeddings'''
    masked = (geo2D[:, 2] > depthMask) & (geo2D[:, 1] < xMask) & (geo2D[:, 0] > yMask)
    geo2D = geo2D[~masked]
    z_grid = np.arange(np.max(geo2D[:, 2]), np.min(geo2D[:, 2]), -1)
    subset = data = pd.read_pickle(output_pred_resampled_pkl)
    x_min, x_max = subset['xmt'].min(), subset['xmt'].max()
    y_min, y_max = subset['ymt'].min(), subset['ymt'].max()
    points_int_x, points_int_y = np.arange(x_min, x_max, scale), np.arange(y_min, y_max, scale)
    xys_toint = list(product(points_int_y, points_int_x))
    yxs = np.array([[m[0], m[1]] for m in xys_toint])
    yxs = np.hstack((yxs,
                     np.array([np.nan]*len(yxs)).reshape(-1, 1),
                     np.array([np.nan]*len(yxs)).reshape(-1, 1)))

    recake = np.zeros(shape=(z_grid.shape[0],
                             np.arange(y_min, y_max, scale).shape[0],
                             np.arange(x_min, x_max, scale).shape[0]))
    for i, n in enumerate(z_grid):
        data3 = geo2D[np.where((geo2D[:, 2] > n - 0.5) & (geo2D[:, 2] <= n + 0.5))]
        print(i, data3.shape[0])
        sort_indexs = np.lexsort((data3[:, 1], data3[:, 0]))
        data4 = data3[sort_indexs]
        indexes = np.apply_along_axis(find_idxs, 1, data4[:, 0:2], yxs[:, 0:2])
        yxs[indexes, 2], yxs[indexes, 3] = data4[:, 2], data4[:, 3]
        recake[i, :, :] = yxs[:, 3].reshape(np.arange(y_min, y_max, scale).shape[0],
                                            np.arange(x_min, x_max, scale).shape[0])
    return recake


####stopped here


def random_color(feature):
    return {
        'color': 'black',
        'fillColor': random.choice(['red', 'yellow', 'green', 'orange','blue','cyan','magenta']),
    }

litho_classes = {0: 'alluvium', 1: 'bedrock', 2: 'carbonaceous',
                 3: 'cavity', 4: 'chemical', 5: 'coarse sediments', 6: 'conglomerate',
                 7: 'fine sediments', 8: 'intrusive', 9: 'limestone', 10: 'metamorphic',
                 11: 'peat', 12: 'sandstone', 13: 'sedimentary', 14: 'shale',
                 15: 'soil', 16: 'volcanic', 17: 'water', 18: 'empty'}

litho_colors = {0: (0.2, 0.15, 0.05, 0.4), 1: (0.8, 0.8, 0.8, 0.7),
                2: (0.3, 0.4, 0.05, 0.6), 3: (0.8, 0.9, 0.1, 0.2),
                4: (0.4, 0.4, 0.9, 0.6), 5: (0.7, 0.7, 0.5, 1),
                6: (0.1, 0.2, 0.1, 0.7), 7: (0.54, 0.3, 0.04, 1),
                8: (0.9, 0.3, 0.2, 0.6), 9: (0.85, 0.96, 0.1, 0.6),
                10: (0.95, 0.8, 0.1, 0.7), 11: (0.25, 0.8, 0.3, 0.4),
                12: (0.9, 0.06, 0.5, 0.6), 13: (0.2, 0.35, 0.4, 0.6),
                14: (0.35, 0.45, 0.5, 0.7), 15: (0.5, 0.5, 0.2, 0.8),
                16: (0.45, 0.36, 0.46, 0.7), 17: (0.4, 0.8, 0.99, 0.4),
                18: (1, 1, 1, 0)}

classes = ['alluvium', 'bedrock', 'carbonaceous', 'cavity', 'chemical',
           'coarse sediments', 'conglomerate', 'fine sediments', 'intrusive',
           'limestone', 'metamorphic', 'peat', 'sandstone', 'sedimentary',
           'shale', 'soil', 'volcanic', 'water']

cols = [litho_colors[n] for n in range(18)]
patches = [mpatches.Patch(color=litho_colors[n], label=litho_classes[n])
           for n in range(18)]

def preprocess_3Ddata(geo3D_path):
    recake = np.load(geo3D_path)
    recake1 = recake[::-1, ::-1, ::-1]
    recake1 = recake1[:, :, :]
    recake1[np.isnan(recake1)] = 18
    recake1[recake1 == 0] = 18
    classes_array = recake1.astype('int16')
    return classes_array

def preprocess_2Ddata(geo2D_path, depthMask=np.nan, xMask=np.nan, yMask=np.nan):
    data1 = np.load(geo2D_path)
    masked = (data1[:, 2] > depthMask) & (data1[:, 1] < xMask) & (data1[:, 0] > yMask)
    data_sliced = data1[~masked]
    return data_sliced

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
                        dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot3D(x, y, z, unprocessed2D, processed2D, filled, facecolors, title, figsize=(10, 8), elevation=30, azimuth=320):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elevation, azimuth)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel('x', fontsize=16)
    ax.set_xticks(np.linspace(np.min(x), np.max(x), 4))
    ax.set_xticklabels(np.round(np.linspace(np.max(unprocessed2D[:, 1]),
                                            np.min(unprocessed2D[:, 1]),
                                            4), 1))
    ax.set_ylabel('y', fontsize=16)
    ax.set_yticks(np.linspace(np.min(y), np.max(y), 4))
    ax.set_yticklabels(np.round(np.linspace(np.max(unprocessed2D[:, 0]),
                                            np.min(unprocessed2D[:, 0]),
                                            4), 1))
    ax.set_zlabel('z', fontsize=16)
    ax.set_zticks(np.linspace(np.min(z)-80, np.max(z)+80, 4))
    ax.set_zticklabels(np.round(np.linspace(np.min(processed2D[:, 2])-80*((np.max(processed2D[:, 2])-np.min(processed2D[:, 2]))/(np.max(z)-np.min(z))),
                                np.max(processed2D[:, 2])+80*((np.max(processed2D[:, 2])-np.min(processed2D[:, 2]))/(np.max(z)-np.min(z))),
                                4), 1))
    ax.set_zlim3d(np.min(z)-80, np.max(z)+80)
    ax.voxels(x, y, z, filled, facecolors=facecolors)
    ax.set_title(title, fontsize=18, loc='left', fontweight='bold')
    fig.tight_layout()
    return fig

def entropy(probabilities):
    return np.nansum([n * np.log(n) for n in probabilities])*-1/np.log(18)

def get_entropies(embeddings_path, model):
    '''Function to create entropies from embeddings
    Inputs:
        -embeddings_path: dataframe with embeddings
        -model: mlp model trained
    Outputs:
        -ent_mean: mean entropy per lithological class
        -quantity: quantity (%) of lithological class'''
    ver = pd.read_pickle(embeddings_path)
    probs = model.predict(np.array(ver['mean'].tolist()))
    ver['probs'] = probs.tolist()
    ver['entropy'] = ver['probs'].apply(lambda x: entropy(x))
    ent_mean = [ver[ver['pred'] == n].entropy.mean() for n in range(18)]
    quantity = [ver[ver['pred'] == n].entropy.count()/len(ver)*100 for n in range(18)]
    return ent_mean, quantity

def classification_entropies(mean_entropies, quantities, colors, patches, classes):
    '''Function to plot uncertainties and quantities for lithological classes
    Intputs:
        -mean_entropies: mean entropy per lithological class
        -quantities: quantity (%) of lithological class
        -colors: colors for lithologies
        -patches: patches for lithologies
        -classes: labels for lithological classes
    Outputs:
        -fig: barplot with uncertainties and quantities for lithologies'''
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(classes, mean_entropies, color=cols, label='NSW',
              edgecolor='black')
    ax[0].set_ylim(0, 1)
    ax[0].tick_params(axis='y', labelsize=10)
    ax[0].set_ylabel('Mean entropy', fontsize=13)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False,
                      labelbottom=False)
    ax[1].bar(classes, quantities, color=colors, label='NSW',
              edgecolor='black')
    ax[1].tick_params(axis='y', labelsize=10)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False,
                      labelbottom=False)
    ax[1].set_ylabel('Lithological classes (%)', fontsize=13)
    lgd = fig.legend(handles=patches, ncol=6,
                     fontsize='small', bbox_to_anchor=(0.84, 0.17),
                     fancybox=True)
    fig.subplots_adjust(bottom=0.7)
    fig.tight_layout()
    return fig

def CIfunction(arr):
    sort_indices = np.argsort(arr)
    return 1 - (arr[sort_indices[-1]] - arr[sort_indices[-2]])

def train_vali(subset):
    grouped_DF = subset.groupby(['x', 'y'])
    groups = np.arange(grouped_DF.ngroups)
    np.random.seed(0)
    sampling = np.random.choice(groups,
                                size=round(0.1*grouped_DF.ngroups),
                                replace=False)
    test = subset[grouped_DF.ngroup().isin(sampling)]
    others = subset[~grouped_DF.ngroup().isin(sampling)]
    return test, others

def uncertainties2D(subset, repetitions, model, raster_path):
    src = gdal.Open(raster_path)
    geo = src.GetGeoTransform()
    rb = src.GetRasterBand(1)
    x_min, x_max = subset.x.min(), subset.x.max()
    y_min, y_max = subset.y.min(), subset.y.max()
    z_min, z_max = subset.z.min(), subset.z.max()
    points_int_x, points_int_y = np.arange(x_min, x_max, 100), np.arange(y_min, y_max, 100)
    xys_toint = list(product(points_int_y, points_int_x))
    test, others = train_vali(subset)
    others_grouped = others.groupby(['x', 'y'])
    others_groups = np.arange(others_grouped.ngroups)
    z_min, z_max = subset.z.min(), subset.z.max()
    lins = []
    for n in range(repetitions):
        train_sampling = np.random.choice(others_groups,
                                          size=round(0.9*others_grouped.ngroups),
                                          replace=False)
        training = others[others_grouped.ngroup().isin(train_sampling)]
        Lnss = []
        for x in np.arange(0, 60, 1):
            de = 1
            z = np.arange(z_min+x-0.5, z_min+x+0.5, de)
            points_train = training[(training.z > z[0]*de) &
                                    (training.z < (z[0]+1)*de)]

            Ln = interpolate.LinearNDInterpolator(points_train[['y', 'x']].values,
                                                  np.array(points_train['mean'].tolist()))
            Ln_int = Ln(xys_toint)
            Lns = model.predict(Ln_int, verbose=0)

            Lns.sort(axis=1)
            CIs = np.apply_along_axis(CIfunction, axis=1, arr=Lns)
            entro = np.apply_along_axis(entropy, axis=1, arr=Lns)

            elev = []
            for m in xys_toint[:]:
                mx, my = m[1], m[0]
                px = int((mx - geo[0])/geo[1])
                py = int((my - geo[3])/geo[5])
                intval = rb.ReadAsArray(px, py, 1, 1)
                elev.append(intval[0][0])

            elev = [e - x - 0.5 for e in elev]
            yxs = np.array([[n[0], n[1]] for n in xys_toint])
            yxzsLn = np.hstack((yxs,
                                np.array(elev).reshape(-1, 1),
                                CIs.reshape(-1, 1),
                                entro.reshape(-1, 1)))
            Lnss.append(yxzsLn)
        Lns1 = np.vstack(Lnss)
        lins.append(Lns1)
        return lins

def CI_points(list_arrays, mask=False):
    yxs = list_arrays[0][:, :3]
    Cis = np.stack([n[:, 3] for n in list_arrays], axis=1)
    ConfusionIndex = np.apply_along_axis(np.nanmean, axis=1, arr=Cis)
    return np.hstack([yxs,
                      ConfusionIndex.reshape(-1, 1)])

def Ent_points(list_arrays, mask=False):
    yxs = list_arrays[0][:, :3]
    Ents = np.stack([n[:, 4] for n in list_arrays], axis=1)
    Entropies = np.apply_along_axis(np.nanmean, axis=1, arr=Ents)
    return np.hstack([yxs,
                      Entropies.reshape(-1, 1)])
