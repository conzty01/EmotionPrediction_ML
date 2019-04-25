import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
import time
import heartrate

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data,axis=0)
    return y

def get_info(val,start,end):
    mean = val.mean()
    sd = np.std(val)
    maxy = val.max()
    miny = val.min()
    rangey = maxy - miny
    slope = (val[-1] - val[0]) / (end-start)
    return pd.DataFrame.from_dict({"mean":[mean],"std":[sd],'range':[rangey],'max':[maxy],'min':[miny],'slope':[slope],"start":[start],"end":[end]})

def orginize(df):
    infodic = {}
    listOfId = []
    listOfMean = []
    listOfSd = []
    listOfLabel = []
    count=0
    for i in range(len(df)//window):
        count+=1
        a = i*window
        b = a + window
        
        info = df[a:b][['y','lables']]
        listOfId.append(count)
        listOfMean.append(info['y'].mean())
        listOfSd.append(info['y'].std())
        listOfLabel.append(info['lables'].mean())
        
    infodic['id'] = listOfId
    infodic['mean'] = listOfMean
    infodic['sd'] = listOfSd
    infodic['labels'] = listOfLabel
    
    return pd.DataFrame.from_dict(infodic) 

def shifingWindow(df,shiftWin, shiftStep):
    returnDf = pd.DataFrame(columns= ['mean','std','range','max','min','slope','label'])
    for i in range(0,len(df),shiftStep):
        maxRange = i + shiftWin
        oneRow = get_info(df[i:maxRange]['y'].values,i,maxRange)
        returnDf = returnDf.append(oneRow.iloc[0],ignore_index=True)
        returnDf['label'] = df['lables'].iloc[0]
    return returnDf

def openPickle(thePickle):
    with open(thePickle, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()
def eda(row,df):
    index = int(row['index'])
    return df.iloc[index*175][0]
def bvp(row,df):
    index = int(row['index'])
    if index*11 <= 4255299:
        #print(index*11)
        return df.iloc[index*11][0]
    else:
        return 0

def getOne(p,one):
    oneDict = p['signal']['chest'][one]
    labels = p['label']
    onedf = pd.DataFrame(data=oneDict).reset_index()
    #labelsDf= pd.DataFrame(data=labels)
    #if one == 'EDA':
        #onedf['lables'] = onedf.apply(lambda row: eda(row,labelsDf),axis=1)
    #else:
        #onedf['lables'] = onedf.apply(lambda row: bvp(row,labelsDf),axis=1)
    onedf['lables'] = pd.Series(labels, index=onedf.index)
    y = butter_lowpass_filter(oneDict, 5, 700, 6)
    onedf['y'] = pd.DataFrame(y)
    #onedf.rename(index=str, columns={"index": "index", 0: "y",'lables':'lables'},inplace=True)
    stressDf = onedf[(onedf['lables'] == 2)]
    amuseDf = onedf[(onedf['lables'] == 3)]  
    return (stressDf,amuseDf)

def graphData(df,i):
    #plt.plot(t, df, 'b-', label='data')

    for test in list(df.columns):
        if "Unnamed" in test or "start" in test or "end" in test or "label" in test or "subject" in test:
            df = df.drop(test,axis=1)
    columns = list(df.columns)    
    for by in columns:
        fig = plt.figure()
        fig.add_axes()        
        temp = df.reset_index()
        num = 101 + ((i%10)*10) + (i//10)*100
        plt.plot(range(len(temp[temp['stress'] == 1])), temp[by][temp['stress'] == 1], 'g-', linewidth=2,label='stress')
        plt.plot(range(len(temp[temp['amuse']==1])), temp[by][temp['amuse'] == 1], 'r-', linewidth=2,label='amuse')
        plt.title('Subject '+str(i)+" "+by)
        plt.xlabel('row')
        plt.ylabel(by)
        plt.savefig('S'+str(i)+'\\'+by+'.png')
        #plt.show()
    plt.xlabel('Time [sec]')
    #plt.xlim([1010 ,1000])


def test():
    for i in range(2,18):
        p = openPickle('S'+str(i)+'.pkl')
        #with open('S2ECG1.csv','w') as f:
            #onedf = pd.DataFrame(data=p['signal']['chest']['ECG'])
            #f.write(onedf.to_csv())
        #p = pd.read_csv('S2ECG.csv')
        #graphData(p,'0')
        #print(p['signal']['chest']['ECG'])
        for j in ['ECG','EDA']:
            stressDf , amuseDf = getOne(p,'ECG')
            #stressDf = pd.read_csv('ECGstress.csv')
            #amuseDf = pd.read_csv('ECGamuse.csv')
            #stressInfo = shifingWindow(stressDf,42000,175)
            #amuseInfo = shifingWindow(amuseDf,420000,175)
            with open('ECGstressraw.csv','w') as f:
                f.write(stressDf.to_csv())
            with open('ECGamuseraw.csv','w') as f:
                f.write(amuseDf.to_csv())
        #print('done')
        #graphData(stressInfo)
        #with open('stressEDA.csv','w') as f:
            #f.write(stressInfo.to_csv())
        #with open('amuseEDA.csv','w') as f:
            #f.write(amuseInfo.to_csv())
def writeFile(name,stress,amuse,subject,final = False):
    if final:
        end = 'final'
    else:
        end = 'raw'
    with open(''+subject+'\\'+subject+''+name+'stress'+end+'.csv','w') as f:
        f.write(stress.to_csv())
    with open(''+subject+'\\'+subject+''+name+'amuse'+end+'.csv','w') as f:
        f.write(amuse.to_csv())    
def main():

    for i in [10]:#[2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]:
        
        #p = openPickle('S'+str(i)+'\S'+str(i)+'.pkl')
        
        #stressEDA , amuseEDA = getOne(p,'EDA')
        #writeFile('EDA',stressEDA,amuseEDA,'S'+str(i))
        #stressInfo = shifingWindow(stressEDA,420,1)
        #amuseInfo = shifingWindow(amuseEDA,420,1)
        #writeFile('EDA',stressInfo,amuseInfo,'S'+str(i),True)
        
        #stressECG, amuseECG = getOne(p,'ECG')
        #writeFile('BVP',stressECG,amuseECG,'S'+str(i))
        df = pd.read_csv('S'+str(i)+'\S'+str(i)+'combined.csv')
        graphData(df,i)
        #rristress = heartrate.findrrv(stressECG)
        #rriamuse = heartrate.findrrv(amuseECG)
        #writeFile('BVP',rristress,rriamuse,'S'+str(i),True)
        print('done with '+str(i))
                  
main()