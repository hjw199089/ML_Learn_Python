'''
机器学习实战-回归
'''

from numpy import *


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stepWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # yMat归一化
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    # xMat归一化
    xMat = regularize(xMat)
    m, n = shape(xMat)
    # 返回迭代后的系数矩阵，每行为一次迭代，最后一行为本地迭代的最优
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T) #打印上次迭代的结果
        lowestError = inf #初识化误差
        for j in range(n): #迭代n个特征
            for sign in [-1,1]: #对每个特征加上或减去步进值对比效果
                wsTest= ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssErr = rssError(yMat.A,yTest.A)
                if rssErr < lowestError:
                    lowestError = rssErr
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

