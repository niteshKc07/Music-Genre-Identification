import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "D:\minor project\MUSIC GENRE CLASSIFICATION\selo1.wav"

#waveform
signal, sr = librosa.load(file, sr= 22050)
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#fft -> spectrum sr*t -> 22050*30(sec)--> more or less this value
#perform foureir transform
fft = np.fft.fft(signal)

#calc abs values on complex number to get magnitude
magnitude = np.abs(fft)

#create freq variable
frequency = np.linspace(0,sr, len(magnitude)) #from 0 to sr
#linspace give evenly space num in interval

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]


plt.plot(left_frequency,left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

#stft(short time fourier trans)-->spectrogram
n_fft = 2048  #no of sample of fft
hop_length = 512 #amount to shift each fourier transform to right

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectrogram,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

#mfccs
mfccs = librosa.feature.mfcc(signal,n_fft=n_fft, hop_length=hop_length, n_mfcc=20)
librosa.display.specshow(mfccs,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()

