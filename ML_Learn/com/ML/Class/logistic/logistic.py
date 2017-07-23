
#logistic 学习笔记

from numpy import *

def loadDataSet(filepath):
    dataMat = [] # 100*3
    labelMat =[]
    fr = open(filepath)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0 ,float(lineArr[0]), float(lineArr[1])]) #补X0=1.0
        labelMat.append(int(lineArr[2])) #1*100
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() #转换成数组和转置100*1
    m,n = shape(dataMatrix)
    alpha = 0.001 #步进系数
    maxCycles  = 500
    weights = ones((n,1))#系数初识化为 1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights



