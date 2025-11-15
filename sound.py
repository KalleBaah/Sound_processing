# Imports
import numpy as np
import wave  
import librosa, librosa.display
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal
import matplotlib.pyplot as plt

# Class for single sound sample (or optional appended samples) and various processing and plotting
# There are only static methods in this class that return results without  making instances of variables

class Sound:
    # Read in raw data file(s) and make array
    @staticmethod
    def readwavfile(files, wavfolder):
        #print(files, wavfolder)
        arrlist = []
        # Read wav file(s)
        for file in files:
            with wave.open(wavfolder + file) as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16)
                #print(arr)
            arrlist.append(arr)
            #self.N += len(arr)
        # Transform to single 1D numpy array
        if len(arrlist) > 1:
            return np.concatenate(arrlist)
        else:
            return arrlist[0]
        
    # Low pass filter  
    @staticmethod
    def lowpass(data, fs, cutoff, poles):
        sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=fs, output='sos')
        return scipy.signal.sosfiltfilt(sos, data)
    
    # High pass filter
    @staticmethod
    def highpass(data, fs, cutoff, poles):
        sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=fs, output='sos')
        return scipy.signal.sosfiltfilt(sos, data)
    
    # Band pass filter
    @staticmethod
    def bandpass(data, fs, edges, poles):
        sos = scipy.signal.butter(poles, edges, 'bandpass', fs=fs, output='sos')
        return scipy.signal.sosfiltfilt(sos, data)
        
    # Calculate RMS
    @staticmethod
    def rms(data):
        return np.sqrt(np.mean(np.square(data.astype(float)))) 
      
    # Calculate crest factor
    @staticmethod
    def crestfactor(data):
        peaklvl = np.amax(np.abs(data))
        rms = Sound.rms(data)
        if rms > 0:
            return peaklvl/rms
        else:
            return 0
    
    # Calculate dtf with fft method (scipy)
    @staticmethod
    def soundfft(data, fs):
        fftarr = fft(data)
        #fqarr = fftfreq(len(data), 1/fs)
        # Do shift if complex
        if np.iscomplex(fftarr).any():
            #print('complex')
            return fftshift(fftarr)
        else:
            return fftarr
        
    # Calculate Mel spectogram (librosa)
    @staticmethod
    def melspec(data, fs, hop, nfft, db=True):
        melarr = librosa.feature.melspectrogram(y=data.astype(np.float32), sr=fs, hop_length=hop, n_fft=nfft)
        spectrogram = np.abs(melarr)
        if db:
            return librosa.power_to_db(spectrogram, ref=np.max)
        else:
            return spectrogram
        
    # Plot spectrogram (librosa)
    @staticmethod
    def plotspec(data, fs, hop, file, label):
        plt.figure(figsize=(8, 7))
        librosa.display.specshow(data, sr=fs, x_axis='time', y_axis='mel', cmap='magma', hop_length=hop)
        plt.colorbar(label='dB')
        plt.title('Mel-Spectrogram (dB) file: '+ file + ' label: ' + str(int(label)))
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()  

    # Plot dft (matplotlib)
    @staticmethod
    def plotdft(data, fs, file, label, half=True):
        # Show half or full
        freq = fftfreq(len(data), 1/fs) 
        if half:
            data = fftshift(data)
            freq = fftshift(freq)
            y = data[:int(len(data)/2)]
            x = freq[int(len(freq)/2):]
        else:
            y = data.astype(np.float32)
            x = freq.astype(np.int16)
        plt.figure(figsize=(8, 6))
        plt.plot(x, (2/len(data))*np.abs(y))
        plt.xticks(np.arange(x[0], x[-1], 1000))
        plt.title('Discrete-Fourier Transform file: '+ file + ' label: ' + str(int(label)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xticks(rotation=45)
        plt.show() 

    # Plot wave file data (librosa)
    @staticmethod
    def plotsound(data, fs, file, label):        
        plt.figure(figsize=(12, 5))
        librosa.display.waveshow(data.astype(np.float32), sr=fs)
        plt.title('Waveplot file: '+ file + ' label: ' + str(int(label)))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()

    # Plot several sound files
    @staticmethod
    def plotmultsound(data, fs, file, label):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        librosa.display.waveshow(data[0].astype(np.float32), sr=fs, ax=ax1)
        librosa.display.waveshow(data[1].astype(np.float32), sr=fs, ax=ax2)
        librosa.display.waveshow(data[2].astype(np.float32), sr=fs, ax=ax3)
        ax1.set(title='Waveplot file: '+ file[0] + ' label: ' + str(int(label[0])))
        ax2.set(title='Waveplot file: '+ file[1] + ' label: ' + str(int(label[1])))
        ax3.set(title='Waveplot file: '+ file[2] + ' label: ' + str(int(label[2])))
        plt.subplots_adjust(hspace=1)
        #plt.xlabel('Time (s)')
        #plt.ylabel('Amplitude')
        plt.show()

    # Plot simple all data 
    @staticmethod
    def plotsimpleall(data, label, xlabel, ylabel):
        plt.figure()
        plt.plot(data)
        plt.title(label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()  





    






