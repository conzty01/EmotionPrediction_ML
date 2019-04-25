import pandas as pd
import numpy as np
def fixingColumns(sub,machine, emotion, wrist = False):
    if wrist:
        temp = pd.read_csv('S'+str(sub)+'\S'+str(sub)+'wrist'+str(machine)+str(emotion)+'final.csv')
        temp.columns = [item + str(machine) + 'wrist' for item in temp.columns]
    else:
        try:
            temp = pd.read_csv('S'+str(sub)+'\S'+str(sub)+str(machine)+str(emotion)+'final.csv')
            temp.columns = [item + str(machine) for item in temp.columns]
        except FileNotFoundError:
            try:
                temp = pd.read_csv('S'+str(sub)+'\S'+str(sub)+str(machine)+str(emotion)+'.csv')
                temp.columns = [item + str(machine) for item in temp.columns]
            except FileNotFoundError:
                temp = pd.DataFrame()
    if wrist:
        if emotion == 'stress':
            temp['stress'] = 1
            temp['amuse'] = 0
        elif emotion == 'amuse':
            temp['stress'] = 0
            temp['amuse'] = 1
        else:
            temp['stress'] = 0
            temp['amuse'] = 0
    if 'Unnamed: 0'+machine in temp.columns:
        temp = temp.drop(['Unnamed: 0'+machine],axis=1)
    if 'Unnamed: 13'+machine in temp.columns:
        temp = temp.drop(['Unnamed: 13'+machine],axis=1)
    return temp
  
  
templist = []
for i in [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]:
    with open('S'+str(i)+'\S'+str(i)+'combined.csv','r') as f:
        templist.append(pd.read_csv(f))
    ECGA = fixingColumns(i,'ECG','amuse')
    ECGS = fixingColumns(i,'ECG','stress')
    EDAA = fixingColumns(i,'EDA','amuse')
    EDAS = fixingColumns(i,'EDA','stress')
    WEDAA = fixingColumns(i,'EDA','amuse',True)
    WEDAS = fixingColumns(i,'EDA','stress',True)
    RESPA = fixingColumns(i,'Resp','amuse')
    RESPS = fixingColumns(i,'Resp','stress')
    TEMPA = fixingColumns(i,'Temp','amuse')
    TEMPS = fixingColumns(i,'Temp','stress')
    EMGA = fixingColumns(i,'EMG','amuse')
    EMGS = fixingColumns(i,'EMG','stress')     
    stressDf = pd.concat([ECGS,EDAS,WEDAS,RESPA,RESPS,TEMPA,TEMPS,EMGA,EMGS],axis=1)
    stressDf.dropna(subset = ['meanHrECG'],inplace=True)
    meanStress = stressDf.mean(axis=0)
    stressDf.replace(np.nan,meanStress,inplace=True)
    amuseDf = pd.concat([ECGA,EDAA,WEDAA,RESPA,RESPS,TEMPA,TEMPS,EMGA,EMGS],axis=1)
    amuseDf.dropna(subset = ['meanHrECG'],inplace=True)
    meanAmuse = stressDf.mean(axis=0)
    amuseDf.replace(np.nan,meanAmuse,inplace=True)    
    finalDf = pd.concat([stressDf,amuseDf])
    finalDf = finalDf.dropna().reset_index().drop('index',axis=1)
    finalDf['subject'] = i
    with open('S'+str(i)+'\S'+str(i)+'combined.csv','w') as f:
        f.write(finalDf.to_csv())
with open('alldata.csv','w') as f:
    f.write(pd.concat(templist,sort=False).to_csv())
df = pd.read_csv('alldata.csv')
with open('describe.csv','w') as f:
    f.write(df.describe().to_csv())
    
    
    