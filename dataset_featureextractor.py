import librosa as lb
import librosa.display
import numpy as nd
import csv
from glob import glob

audio_files= glob('C:\minor project\selo\*.wav')
for i in range(len(audio_files)):

    signal,rate=lb.load(audio_files[i])
    signal=nd.array(signal)
    #name
    name=str(audio_files[i])
    name=name.replace("C:\minor project\selo",'')
    #chroma shift
    chroma_shift= librosa.feature.chroma_stft(signal)
    chroma_shift_mean=nd.average(chroma_shift)
    chroma_shift_var=nd.var(chroma_shift)

    #rms
    rms=librosa.feature.rms(signal)
    rms_mean=nd.average(rms)
    rms_var=nd.var(rms)

    #sepctral centroid
    sc=librosa.feature.spectral_centroid(signal)
    sc_mean=nd.average(sc)
    sc_var=nd.var(sc)

    #spectral bandwidth
    sb=librosa.feature.spectral_bandwidth(signal)
    sb_mean=nd.average(sb)
    sb_var=nd.var(sb)

    #spectral rollof
    sr=librosa.feature.spectral_rolloff(signal)
    sr_mean=nd.average(sr)
    sr_var=nd.var(sr)

    #zero crossing rate
    zcr=librosa.feature.zero_crossing_rate(signal)
    zcr_mean=nd.average(zcr)
    zcr_var=nd.var(zcr)

    #harmonics
    harmonics=librosa.effects.harmonic(signal)
    harmonic_mean=nd.average(harmonics)
    harmonic_var=nd.var(harmonics)

    #tempo
    tempo=librosa.beat.tempo(signal)
    tempo=nd.average(tempo)

    #mfcc
    mfccs=librosa.feature.mfcc(signal)
    mfcc_mean=nd.average(mfccs)
    mfcc_var=nd.var(mfccs)

    #mfcc1
    mfcc1=librosa.feature.delta(mfccs, order=1)
    mfcc1_mean=nd.average(mfcc1)
    mfcc1_var=nd.var(mfcc1)

    #mfcc2
    mfcc2=librosa.feature.delta(mfccs, order=2)
    mfcc2_mean=nd.average(mfcc2)
    mfcc2_var=nd.var(mfcc2)

    #writting files in csv
    filename="test.csv"
    if i == 0:
        fields=['name','chroma_stft_mean','chroma_stft_var',	'rms_mean',	'rms_var',	'spectral_centroid_mean',	'spectral_centroid_var',	'spectral_bandwidth_mean',	'spectral_bandwidth_var',	'rolloff_mean',	'rolloff_var',	'zero_crossing_rate_mean',	'zero_crossing_rate_var',	'harmony_mean',	'harmony_var','tempo',	'mfcc_mean','mfcc_var','mfcc1_mean'	,'mfcc1_var'	,'mfcc2_mean'	,'mfcc2_var']
        with open(filename, 'w',newline='') as csvfile:
            csvwriter=csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvfile.close()

    rows=[name,chroma_shift_mean,chroma_shift_var,rms_mean,rms_var,sc_mean,sc_var,sb_mean,sb_var,sr_mean,sr_var,zcr_mean,zcr_var,harmonic_mean,harmonic_var,tempo,mfcc_mean,mfcc_var,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var]
    with open(filename, 'a',newline='') as csvfile:
        csvwriter=csv.writer(csvfile)
        csvwriter.writerow(rows)
