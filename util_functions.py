#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 08:07:29 2020

@author: deepak
"""

#import os
import numpy as np
#from scipy.io import wavfile
from scipy import signal
#import matplotlib.pyplot as plt

class windowing_speech():
    'an iterator that gives the framed segments of speech signal (x)'
    'frame_size and frame_shift should be in samples , x should be single channel'
    def __init__(self,x,frame_size,frame_shift,window=True,MVN=True):
        self.window = window
        self.MVN = MVN
        self.x=x
        self.frame_size = int(frame_size)
        self.frame_shift = int(frame_shift)
        self.st_idx = []
        self.count = 0
        self.total_frames = 0
        self.no_of_frames()
        self.__indices__()
    def no_of_frames(self):
        # +1 needs to be done because indexing starts from 0
        self.total_frames = int((len(self.x)-self.frame_size) // self.frame_shift + 1 )
        return self.total_frames
    def __indices__(self):
        if(self.frame_shift>self.frame_size):
            print("frame shift greater than frame size")
            return
        self.st_idx = [x for x in range(0,len(self.x),self.frame_shift)]
        #print(self.st_idx)
        if (len(self.st_idx) < self.total_frames):
            print("error!!")
            return
        return
    def __window__(self):
        return np.hamming(self.frame_size)
    def __MVN__(self,x):
        x = x-np.mean(x)
        x = x/max(x)
        return x
    def __getframe__(self):
        if(self.count<self.total_frames):
            # <= doesn't come because index is updated at the end of the loop
            x_frame = self.x[self.st_idx[self.count]:self.st_idx[self.count]+self.frame_size]
            x_frame = np.array(x_frame)
            if(self.MVN == True):
                x_frame = self.__MVN__(x_frame)
            if(self.window == True):
                x_frame = np.array(x_frame)*np.array(self.__window__())
            #x_windowed = np.array(x_mvn)*np.array(self.__window__())
            self.count = self.count+1
            return x_frame
        else:
            self.count = 0
            print("count exceeded total frames")
            return
class waveform_features():
    'Gives us basic waveform level features like short term energy, short term zero crossings'
    def __init__(self,x,fs,frame_size,frame_shift):
        self.x = x
        self.fs = fs
        self.frame_size = int(frame_size)
        self.frame_shift = int(frame_shift)
        self.no_of_frames = 0
        #print(isinstance(self.no_of_frames,str))
    def __STE__(self):
        windower = windowing_speech(self.x,self.frame_size,self.frame_shift,False,False)
        no_of_frames = windower.no_of_frames()
        self.no_of_frames = windower.no_of_frames()
        STE = np.zeros(no_of_frames)
        for i in range(no_of_frames):
            frame = windower.__getframe__()
            STE[i] = np.sum(frame*frame)
            if(STE[i]==0):
                print("error! no speech at all!")
        return STE
    def __endpoints__(self,threshold=0.06,buffer=8):
        ste = self.__STE__()
        avg_ste = np.average(ste)
        #check if the frames are energy frames or not
        is_energy = ste >= threshold*avg_ste
        start_idx=0
        for i in range(self.no_of_frames):
            #Check if the next 8 frames are energy frames 
            n_frames_f = is_energy[i+1:i+buffer+1]
            if(np.sum(n_frames_f)==buffer):
                start_idx = i
                break
        end_idx=0
        for j in range(self.no_of_frames):
            #check if the previous 8 frames are energy frames
            n_frames_b = is_energy[len(is_energy)-buffer-j-1:len(is_energy)-j-1]
            if(np.sum(n_frames_b)==buffer):
                end_idx = len(is_energy)-j
                break 
        if(start_idx>=end_idx):
            print("something is not right!")
            print('start_frame:',start_idx,' end_frame:',end_idx)
            #return
        start_point = int(self.frame_shift*start_idx)
        end_point = int(self.frame_shift*end_idx+self.frame_size)
        return start_point,end_point
            
    def __STZCR__(self):
        windower = windowing_speech(self.x,self.frame_size,self.frame_shift,False,True)
        no_of_frames = windower.no_of_frames()
        self.no_of_frames = no_of_frames
        STZCR = np.zeros(no_of_frames)
        for i in range(no_of_frames):
            frame = windower.__getframe__()
            half_rectify = frame > 0
            count = 0
            for k in range(self.frame_size-1):
                if (half_rectify[k] != half_rectify[k+1] ) : count = count+1
            STZCR[i] = count      
        return STZCR
    def __pitch__(self):
        windower = windowing_speech(self.x,self.frame_size,self.frame_shift,True,True)
        no_of_frames = windower.no_of_frames()
        pitch = np.zeros(no_of_frames)
        for i in range(no_of_frames):
            frame = windower.__getframe__()
            x_corr = np.correlate(frame,frame,mode='full') 
            nyq = self.fs*0.5
            f_cutoff = 450/nyq
            b,a = signal.butter(10,f_cutoff,btype='low',analog=False)
            y = signal.lfilter(b,a,x_corr)
            peaks = self.findpeaks(y)
            peakdist_1 = peaks[1]-peaks[0]
            peakdist_2 = peaks[2]-peaks[1]
            #peakdist_3 = peaks[3]-peaks[2]
            #avg_peakdist = (peakdist_1+peakdist_2+peakdist_3)/3
            avg_peakdist = (peakdist_1+peakdist_2)/2
            #avg_peakdist = peakdist_1
            pitch[i] = self.fs//avg_peakdist
        return pitch
    def findpeaks(self,x):
        for i in range(len(x)):
            if x[i]>=0: x[i]=x[i]
            else : x[i]=0
        momentum = 0
        peaks=[]
        idx = 0
        while momentum == 0:
            if x[idx+1] > x[idx]: 
                momentum = 1
            elif x[idx+1] < x[idx]: 
                momentum = -1
            idx=idx+1
        while idx<len(x)-1:
            if x[idx+1] >= x[idx]:
                new_momentum = 1
            else :
                new_momentum = -1
            if new_momentum != momentum and x[idx+1]!=0:
                peaks.append(idx)
            momentum = new_momentum
            idx = idx+1
        return peaks


    
