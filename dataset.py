# Imports
import numpy as np
import wave  
import librosa, librosa.display
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal
import matplotlib.pyplot as plt

# The class with static methods for individual processing
from sound import Sound

# Class that structures the datasets and for a collection of sound samples
# Calls are made to the Sound() class in sound.py

class Dataset():
    def __init__(self, files, wavfolder, fs, labels):
        self.files = files
        self.wavfolder = wavfolder
        self.fs = fs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file = []
        file.append(self.files[idx])
        soundarr = Sound.readwavfile(file, self.wavfolder)
        return soundarr, self.labels[idx], self.files[idx]
    
    def getfs(self):
        return self.fs