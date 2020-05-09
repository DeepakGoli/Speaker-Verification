#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 20:40:35 2020

@author: deepak

Text Independent Speaker verification
"""

import os
import numpy as np
import util_functions
import spectral_features
from scipy.io import wavfile
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization


path_to_database = '/Users/deepak/Desktop/speech_data/spks/'
spks = os.listdir(path_to_database)

def tdfeatures(features):
    '''time delay features'''
    tdfeats = []
    for i in range(len(features)-1):
        tdfeats.append(np.append(features[i],features[i+1]))
    return tdfeats

user=['deepak']
class_1 = []
class_0 = []
td_c1 = []
td_c0 = []

for spk in spks:
    if (spk != '.DS_Store'):
        speech_files = os.listdir(path_to_database+spk)
        for file in speech_files:
            if(file != '.DS_Store'):
                feats = []
                fs,x = wavfile.read(path_to_database+spk+'/'+file)
                
                trim = util_functions.waveform_features(x,fs,fs*0.02,fs*0.01)
                start_point,end_point = trim.__endpoints__(threshold=0.06,buffer=8)
                x = x[start_point:end_point]
                
                windower = util_functions.windowing_speech(x,fs*0.02,fs*0.01)
                no_of_frames = windower.no_of_frames()
                for i in range(no_of_frames):
                    x_win=windower.__getframe__()
                    x_feats = spectral_features.mfcc(x_win,fs)
                    mfcc_features = x_feats.get_mfcc(13)
                    feats.append(mfcc_features[1:])
                    if spk in user:
                        class_1.append(mfcc_features[1:])
                    else:
                        class_0.append(mfcc_features[1:])
                td_feats=np.array(tdfeatures(feats))
                if spk in user:
                    if len(td_c1) == 0:
                        td_c1 = td_feats
                    else:
                        td_c1 = np.concatenate((td_c1,td_feats))
                else:
                    if len(td_c0) == 0:
                        td_c0 = td_feats
                    else:
                        td_c0 = np.concatenate((td_c0,td_feats))
                   
##############################################################################
                        
''' Feed forward neural network as classifier'''

def custom_loss(y_true,y_pred):
    #Weighted Binary Cross entropy loss function
    w_spk = 0.7
    w_imposter = 0.3
    y_t=k.flatten(y_true)
    y_p=k.flatten(y_pred)
    smooth = 1e-8
    return -k.mean(w_spk*y_t*k.log(y_p+smooth)+w_imposter*(1-y_t)*k.log(1-y_p+smooth))

def speaker_verifier(input_dim):
    model = Sequential()
    model.add(Dense(32,activation='relu',input_dim=input_dim))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model

def td_speaker_verifier(input_dim):
    model = Sequential()
    model.add(Dense(32,activation='relu',input_dim=input_dim))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(2,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model
    
my_spkr = speaker_verifier(12)

#my_spkr.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
my_spkr.compile(optimizer='adam',loss=custom_loss,metrics=['accuracy'])

features = np.append(class_0,class_1,axis=0)
labels = np.append(np.zeros(len(class_0)),np.ones(len(class_1)))

x,y = shuffle(features,labels)

my_spkr.fit(x,y,epochs=200,batch_size=32)


'''time delay nn'''
'''
td_spkr = td_speaker_verifier(24)
td_spkr.compile(optimizer='adam',loss=custom_loss,metrics=['accuracy'])
td_features=np.append(td_c0,td_c1,axis=0)
td_labels = np.append(np.zeros(len(td_c0)),np.ones(len(td_c1)))

td_x,td_y = shuffle(td_features,td_labels)
td_spkr.fit(td_x,td_y,epochs=200,batch_size=64)
'''
##############################################################################


''' Testing '''


path_to_testdatabase = '/Users/deepak/Desktop/speech_data/test_AA1/'
spks = os.listdir(path_to_testdatabase)

user=['deepak']

test_1=[]
test_0=[]

for spk in spks:
    if (spk != '.DS_Store'):
        print(spk)
        speech_files = os.listdir(path_to_testdatabase+spk)
        for file in speech_files:
            test_11=[]
            test_00=[]
            if(file != '.DS_Store'):
                fs,x = wavfile.read(path_to_testdatabase+spk+'/'+file)
                
                trim = util_functions.waveform_features(x,fs,fs*0.02,fs*0.01)
                start_point,end_point = trim.__endpoints__(threshold=0.06,buffer=8)
                x = x[start_point:end_point]
                
                windower = util_functions.windowing_speech(x,fs*0.02,fs*0.01)
                no_of_frames = windower.no_of_frames()
                for i in range(no_of_frames):
                    x_win=windower.__getframe__()
                    x_feats = spectral_features.mfcc(x_win,fs)
                    mfcc_features = x_feats.get_mfcc(13)
                    if spk in user:
                        test_11.append(mfcc_features[1:])
                    else:
                        test_00.append(mfcc_features[1:])
    
            if spk in user:
                test_1.append(test_11)
                score = my_spkr.predict(np.array(test_11))
                print(np.sum(score>=0.5)/len(score))
                #td_test_11 = tdfeatures(test_11)
                #td_score = td_spkr.predict(np.array(td_test_11))
                #print(np.sum(td_score>=0.5)/len(td_score))
            else:
                test_0.append(test_00)
                score = my_spkr.predict(np.array(test_00))
                print(np.sum(score>=0.5)/len(score))
                #td_test_00 = tdfeatures(test_00)
                #td_score = td_spkr.predict(np.array(td_test_00))
                #print(np.sum(td_score>=0.5)/len(td_score))


path_to_testdatabase = '/Users/deepak/Desktop/speech_data/test_spks/'
spks = os.listdir(path_to_testdatabase)

user=['deepak']

test_1=[]
test_0=[]

print('*********different text***********')

for spk in spks:
    if (spk != '.DS_Store'):
        print(spk)
        speech_files = os.listdir(path_to_testdatabase+spk)
        for file in speech_files:
            test_11=[]
            test_00=[]
            if(file != '.DS_Store'):
                fs,x = wavfile.read(path_to_testdatabase+spk+'/'+file)
                
                trim = util_functions.waveform_features(x,fs,fs*0.02,fs*0.01)
                start_point,end_point = trim.__endpoints__(threshold=0.06,buffer=8)
                x = x[start_point:end_point]
                
                windower = util_functions.windowing_speech(x,fs*0.02,fs*0.01)
                no_of_frames = windower.no_of_frames()
                for i in range(no_of_frames):
                    x_win=windower.__getframe__()
                    x_feats = spectral_features.mfcc(x_win,fs)
                    mfcc_features = x_feats.get_mfcc(13)
                    if spk in user:
                        test_11.append(mfcc_features[1:])
                    else:
                        test_00.append(mfcc_features[1:])
            if spk in user:
                test_1.append(test_11)
                score = my_spkr.predict(np.array(test_11))
                print(np.sum(score>=0.5)/len(score))
            else:
                test_0.append(test_00)
                score = my_spkr.predict(np.array(test_00))
                print(np.sum(score>=0.5)/len(score))
