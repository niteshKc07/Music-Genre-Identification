import csv
import math
import librosa
import numpy as nd
genre=['adhunik','bhajan','dohari','selo']

def classifier(filename):
    signal,rate=librosa.load(filename)
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
    diff=[]
    filename='test.csv'
    with open(filename,'r', newline='') as csvfile:
        csvreader=csv.reader(csvfile, delimiter=',')
        line=0
        table=0
        for row in csvreader:
            table=0
            if line==0:
                line=line+1
            elif line>=1:
                line=line+1
                for column in row:
                    if table==0:
                        table=table+1
                    elif table==1:
                        temp_csm=float(column)
                        diff1=math.pow((chroma_shift_mean-temp_csm),2)
                        diff.append(diff1)
                        table=table+1
                    elif table==2:
                        temp_csv=float(column)
                        diff2=math.pow((chroma_shift_var-temp_csv),2)
                        diff.append(diff2)
                        table=table+1
                    elif table==3:
                        temp_rmsm=float(column)
                        diff3=math.pow((rms_mean-temp_rmsm),2)
                        diff.append(diff3)
                        table=table+1
                    elif table==4:
                        temp_rmsv=float(column)
                        diff4=math.pow((rms_var-temp_rmsv),2)
                        diff.append(diff4)
                        table=table+1
                    elif table==5:
                        temp_scm=float(column)
                        diff5=math.pow((sc_mean-temp_scm),2)
                        diff.append(diff5)
                        table=table+1
                    elif table==6:
                        temp_scv=float(column)
                        diff6=math.pow((sc_var-temp_scv),2)
                        diff.append(diff6)
                        table=table+1
                    elif table==7:
                        temp_sbm=float(column)
                        diff7=math.pow((sb_mean-temp_sbm),2)
                        diff.append(diff7)
                        table=table+1
                    elif table==8:
                        temp_sbv=float(column)
                        diff8=math.pow((sb_var-temp_sbv),2)
                        diff.append(diff8)
                        table=table+1
                    elif table==9:
                        temp_srm=float(column)
                        diff9=math.pow((sr_mean-temp_srm),2)
                        diff.append(diff9)
                        table=table+1
                    elif table==10:
                        temp_srv=float(column)
                        diff10=math.pow((sr_var-temp_srv),2)
                        diff.append(diff10)
                        table=table+1
                    elif table==11:
                        temp_zcrm=float(column)
                        diff11=math.pow((zcr_mean-temp_zcrm),2)
                        diff.append(diff11)
                        table=table+1
                    elif table==12:
                        temp_zcrv=float(column)
                        diff12=math.pow((zcr_var-temp_zcrv),2)
                        diff.append(diff12)
                        table=table+1
                    elif table==13:
                        temp_hm=float(column)
                        diff13=math.pow((harmonic_mean-temp_hm),2)
                        diff.append(diff13)
                        table=table+1
                    elif table==14:
                        temp_hv=float(column)
                        diff14=math.pow((harmonic_var-temp_hv),2)
                        diff.append(diff14)
                        table=table+1
                    elif table==15:
                        temp_tm=float(column)
                        diff15=math.pow((tempo-temp_tm),2)
                        diff.append(diff15)
                        table=table+1
                    elif table==16:
                        temp_mfccm=float(column)
                        diff16=math.pow((mfcc_mean-temp_mfccm),2)
                        diff.append(diff16)
                        table=table+1
                    elif table==17:
                        temp_mfccv=float(column)
                        diff17=math.pow((mfcc_var-temp_mfccv),2)
                        diff.append(diff17)
                        table=table+1
                    elif table==18:
                        temp_mfcc1m=float(column)
                        diff18=math.pow((mfcc1_mean-temp_mfcc1m),2)
                        diff.append(diff18)
                        table=table+1
                    elif table==19:
                        temp_mfcc1v=float(column)
                        diff19=math.pow((mfcc1_var-temp_mfcc1v),2)
                        diff.append(diff19)
                        table=table+1
                    elif table==20:
                        temp_mfcc2m=float(column)
                        diff20=math.pow((mfcc2_mean-temp_mfcc2m),2)
                        diff.append(diff10)
                        table=table+1
                    elif table==21:
                        temp_mfcc2v=float(column)
                        diff21=math.pow((mfcc2_var-temp_mfcc2v),2)
                        diff.append(diff21)
                        table=table+1
    size=int(len(diff)/21)
    distance=[]
    for i in range(0,size):
        dis=0
        for j in range(0,21):
            dis=dis+diff[i*21+j]
        square=math.sqrt(dis)
        distance.append(square)
    no_of_songs_indataset=10 

    l_distance=min(distance)
    position=1
    for i in range(len(distance)):
        if distance[i]==l_distance:
            print(position)
            break
        else:
            position=position+1
    if position<=1*no_of_songs_indataset:
        return(genre[0])
    elif position<=2*no_of_songs_indataset:
        return(genre[1])
    elif position<=3*no_of_songs_indataset:
        return(genre[2])
    elif position<=4*no_of_songs_indataset:
        return(genre[3])
