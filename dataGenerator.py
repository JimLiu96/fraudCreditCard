import pandas as pd
import random
import numpy as np

def load_data(filename = 'creditcard.csv', ignoreTime = False):
    data = pd.read_csv(filename)
    numIndex = len(data.columns)
    featureVec = data.iloc[:,:numIndex-1]
    labelVec = data.iloc[:,numIndex-1]
    return featureVec.values, labelVec.values

def resampling(feature,label,alpha=1,method='under_sampling'):
    numInstances = len(label)
    posNum = 0
    negNum = 0
    featureNum = feature.shape[1]
    for insIdx in xrange(numInstances):
        if label[insIdx] > 0:
            posNum += 1
        else:
            negNum += 1
    posIdx = np.where(label == 1)[0]
    negIdx = np.where(label == 0)[0]
    negSampleNum = int(alpha * posNum)
    totalSampleNum = negSampleNum + posNum
    returnArr = np.zeros([totalSampleNum, featureNum])
    returnLabel = np.append(np.ones(posNum),np.zeros(negSampleNum))
    negIdxOri = negIdx[np.random.randint(negNum, size=negSampleNum)]
    for returnIdx in xrange(totalSampleNum):
        if returnIdx < posNum:
            returnArr[returnIdx,:] = feature[posIdx[returnIdx],:]
        else:
            returnArr[returnIdx,:] = feature[negIdxOri[returnIdx-posNum],:]
    return returnArr, returnLabel
