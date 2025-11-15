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

# Class that perform the actual calculations for data in datasets input
# This class also outputs the good, broken and heavyload in a combined numpy array, including labels

class Calculate():
    def __init__(self, datasets):
        self.datasets = datasets
        self.sounds = []
        self.labels = []
        self.files = []
        # Make list of sound array(s) with labels
        for ds in self.datasets:
            for idx in range(len(ds)):
                d, l, f = ds[idx]
                self.sounds.append(d)
                self.labels.append(l)
                self.files.append(f)

    # Get copy of soundarrays as list    
    def getsound(self):
        return np.array(self.sounds), np.array(self.labels), self.files

    # Calculate dft with fft for all arrays
    def calcfft(self, fpar=[44100, 2000, 3], filt=False):
        ffts = []
        for sound in self.sounds:
            if filt:
                sound = Sound.lowpass(sound, fpar[0], fpar[1], fpar[2])
            fft = Sound.soundfft(sound, self.datasets[0].getfs())
            ffts.append(fft)
        return np.array(ffts), np.array(self.labels), self.files
    
    # Calculate mel spec for all arrays
    def calcspec(self):
        specs = []
        for sound in self.sounds:
            spec = Sound.melspec(sound, self.datasets[0].getfs(), hop=1024, nfft=2048)
            specs.append(spec)
        return np.array(specs), np.array(self.labels), self.files
    
    # Calculate rms for all arrays
    def calcrms(self):
        rmses = []
        for sound in self.sounds:
            rms = Sound.rms(sound)
            rmses.append(rms)
        return np.array(rmses), np.array(self.labels), self.files
    
    # Calculate crest factor for all arrays
    def calccrest(self):
        crests = []
        for sound in self.sounds:
            crest = Sound.crestfactor(sound)
            crests.append(crest)
        return np.array(crests), np.array(self.labels), self.files

