#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:19:24 2020

@author: deepak
"""

import numpy as np
from scipy.fftpack import dct
class mfcc():
    'returns mfcc features of a frame'
    def __init__(self,x,fs,preemphasis=True):
        self.x=x
        self.fs = fs
        if preemphasis:
            x_1 = np.append([0],x[:-1])
            self.x = x-0.98*x_1
    def hz2mel(self,hz):
        return 2595*np.log10(1+hz/700)
    def mel2hz(self,mel):
        return 700*(10**(mel/2595)-1)
    def mel_fbank(self,f_low,f_high,n_filt,N_FFT):
        mel_points = np.linspace(self.hz2mel(f_low),self.hz2mel(f_high),n_filt+2)
        hz_points = self.mel2hz(mel_points)
        fft_bin = np.floor((N_FFT+1)*hz_points/self.fs)
        fbank = np.zeros([n_filt,N_FFT//2+1])
        for j in range(0,n_filt):
            for i in range(int(fft_bin[j]),int(fft_bin[j+1])):
                fbank[j,i] = (i-fft_bin[j])/(fft_bin[j+1]-fft_bin[j])
            for i in range(int(fft_bin[j+1]),int(fft_bin[j+2])):
                fbank[j,i] = (fft_bin[j+2]-i)/(fft_bin[j+2]-fft_bin[j+1])
        return fbank
    
    def get_mfcc(self,n_feats=13,N_FFT=512,f_low=0,f_high=None,n_filt=26):
        f_high = f_high or self.fs/2
        x_fft = np.fft.fft(self.x,N_FFT)
        x_psd = np.abs(x_fft[:N_FFT//2+1])**2/N_FFT  
        x_energy = 2*np.sum(x_psd)
        fbank = self.mel_fbank(f_low,f_high,n_filt,N_FFT)
        x_mel = np.dot(x_psd,fbank.T)
        x_mel = np.where(x_mel==0,np.finfo(float).eps,x_mel)
        x_log = np.log(x_mel)
        x_dct = dct(np.reshape(x_log,[n_filt,1]),type=2,axis=1,norm='ortho')
        return np.append(np.log(x_energy),x_dct[1:n_feats])