import scipy.signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hrv.rri import RRi
from hrv.io import read_from_csv
import neurokit as nk
from hrv.classical import time_domain
def nothing():
    for i in range(len(peakTimes)-1):
        if i != len(peakTimes):
            temp = peakTimes[i+1] - peakTimes[i]
            if temp <= 0 :
                print(temp,peakTimes[i+1],peakTimes[i])
            else:
                intervals.append(temp)    
def get_hrv(hrv,hr = False):
    returnhrDic = {}
    returnhrvDic = {}
    
    returnhrDic['meanHr'] = hrv['mean']['hr']
    returnhrDic['medianHr'] = hrv['median']['hr']
    returnhrDic['stdHr'] = hrv['std']['hr']
    returnhrDic['varHr'] = hrv['var']['hr']
    
    
    returnhrvDic['meanHrv'] = hrv['mean']['rri']
    returnhrvDic['medianHrv'] = hrv['median']['rri']
    returnhrvDic['stdHrv'] = hrv['std']['rri']
    returnhrvDic['varHrv'] = hrv['var']['rri']
    
    return returnhrDic,returnhrvDic
        
def findrrv(p):
    samplingRate = 700
    shiftStep = 175
    returnDf = pd.DataFrame()
    count = 0
    for i in range(0,len(p),shiftStep):
        try:
            bio = nk.ecg_preprocess(ecg=p[i:i+(samplingRate*60)][0], sampling_rate=samplingRate)
            #hrv = nk.bio_ecg.ecg_hrv(rpeaks=bio['ECG']['R_Peaks'], sampling_rate=sampling_rate)
            print(bio['ECG']['R_Peaks'])
            #peakTimes = list(scipy.signal.find_peaks(p[i:i+(samplingRate*60)]['y'],100,distance = 25))[0]
            #print(peakTimes)
            peakTimes = bio['ECG']['R_Peaks']
            peakTimes = np.diff(peakTimes)
            peakTimes = peakTimes/samplingRate
            peakTimes = peakTimes.astype(float)
            hrv = RRi(peakTimes)
            #print(hrv.describe())
            hrdic,hrvdic = get_hrv(hrv.describe())
            time = time_domain(hrv)
            returnDic = dict(hrdic,**hrvdic)
            returnDic.update(time)
            temp = pd.DataFrame(returnDic,index=[i])
            returnDf = returnDf.append(temp,ignore_index=True)
            #hrv = nk.bio_ecg.ecg_hrv(rri=peakTimes, sampling_rate=sampling_rate,hrv_features=['time'])   
            count += 1
        except Exception as inst:
            print(p[i:i+(samplingRate*60)][0])
            print(inst)
    print(count)
    return returnDf

    
def graphData(df,by='mean'):
    t = np.linspace(0, len(df)/700, len(df), endpoint=False)
    #plt.plot(t, df, 'b-', label='data')
    plt.plot(t, df[by], 'g-', linewidth=2,label='filtered data')
    plt.xlabel('Time [sec]')
    #plt.xlim([1010 ,1000])
    plt.show()


def main():
    p = openPickle('S2/S2.pkl')
    #with open('S2ECG.csv','w') as f:
        #onedf = pd.DataFrame(data=p['signal']['chest']['ECG'])
        #f.write(onedf.to_csv())
    #rri = pd.read_csv('S2ECGInfo.csv')
    p = pd.read_csv('ECGstressraw.csv')
    rri = findrrv(p)
    #rri.reset_index(inplace=True)
    with open('ECGFinalStress.csv','w') as f:
        f.write(rri.to_csv())    
    #results = time_domain(rri)
    #print(results)    
    #rri.hist(hr=True)
    #graphData(pd.DataFrame(data = rrv[1]),'peak_heights')
#main()