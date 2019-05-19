import pickle
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, freqz, find_peaks
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def getData(sID):
    fileStr = "WESAD/WESAD/S{}/S{}.pkl".format(sID,sID)
    file = open(fileStr,'rb')
    p = pickle._Unpickler(file)
    p.encoding = 'latin1'
    u = p.load()
    data = u['signal']['chest']['Resp']
    labels = u['label']

    df = pd.DataFrame(data=data)
    df['label'] = pd.Series(labels, index=df.index)
    y = butter_lowpass_filter(data, cutoff, fs, order) # Filter data
    #y = data                                            # Don't filter data
    df['y'] = pd.DataFrame(y)
    stressDF = df[(df['label'] == 2)]

    return stressDF["y"].values

def plotData(y):
    T = 6079.0         # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)

    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 2)
    #plt.plot(t, data, 'b-', label='data')
    plt.plot(range(len(y)), y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.ylim()
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()

# Filter requirements.
order = 6
fs = 700.0      # sample rate, Hz
cutoff = 5  # desired cutoff frequency of the filter, Hz

# Create new data columns from the previous data
# Mean, Standard Deviation, Max, Min, Range, Slope
# id, subjID, time, Mean, Standard Deviation, Max, Min, range, slope
# 1, 2, 60, ...
# 2, 2, 120, ...
# 3, 2, 180, ...

timeFrame = 60 # 60 Seconds
timeShift = 0.25 # Shift the timeFrame up by 0.25 seconds

subjectIDs = [2]#,3,4,5,6,7,8,9,10,11,13,14,15,16,17]


# Iterate through all the subject data
for sID in subjectIDs:
    indID = 1
    outFile = open("WESAD/WESAD/S{}/S{}Respstress.csv".format(sID,sID), "w")
    outFile.write(",mean,standard_deviation,max,min,range,slope,breath_rate,in_ex_ratio,in_mean,in_std,ex_mean,ex_std,\n")

    data = getData(sID)
    #plotData(data)
    print(sID, data)

    start_t = 0
    end_t = int(fs*timeFrame)
    step = int(fs*timeShift)
    while end_t <= data.size:
        time_seg = data[start_t:end_t]

        sMean = time_seg.mean()
        sStd = np.std(time_seg)
        sMin = time_seg.min()
        sMax = time_seg.max()
        sRange = sMax - sMin
        sSlope = (time_seg[-1] - time_seg[0]) / (end_t - start_t)

        iPeaks = find_peaks(time_seg,distance=1000,height=1)
        iPeakHeights = iPeaks[-1]["peak_heights"]
        #print(len(time_seg), time_seg)
        print(len(iPeaks[0]),iPeaks)

        pRate = len(iPeaks[0]) / timeFrame
        pMean_In = iPeakHeights.mean()
        pStd_In = np.std(iPeakHeights)
        num_In = len(iPeakHeights)
        # Invert the data (look at exhales)

        for pos in range(len(time_seg)):
            time_seg[pos] = time_seg[pos] * -1
        ePeaks = find_peaks(time_seg, distance=1000, height=1)
        ePeakHeights = ePeaks[-1]["peak_heights"]
        pStd_Ex = np.std(ePeakHeights)
        pMean_Ex = ePeakHeights.mean()
        num_Ex = len(ePeakHeights)

        pRatio = num_In / num_Ex

        outFile.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(indID,sMean,sStd,sMax,sMin,sRange,sSlope,pRate,pRatio,pMean_In,pStd_In,pMean_Ex,pStd_Ex))

        start_t += step
        end_t += step
        indID += 1

    #plotData(data)
    outFile.close()
