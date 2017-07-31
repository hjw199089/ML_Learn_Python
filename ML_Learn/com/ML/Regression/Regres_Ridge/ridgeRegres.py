'''
机器学习实战-回归
'''

from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))  # 获取数据部分
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))  # 获取输出部分
    return dataMat, labelMat


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    demon = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(demon) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = demon.I * (xMat.T * yMat)
    return ws


# a = [1,2,3]
# b = [[2,2,3],[3,3,3],[1,2,3]]
# bMat = mat(b)
# print(bMat)
# print(mean(bMat,0))
# 输出，计算每一维度下的均值
# # [[2 2 3]
# #  [3 3 3]
# #  [1 2 3]]
# # [[ 2.          2.33333333  3.        ]]
def ridgeRegresTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # 减去均值
    # 归一化X
    xMeans = mean(xMat, 0)  # 计算每一维度下的均值
    xVar = var(xMat, 0)  # 计算每一维度下的方差
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30  # 计算30次不同的lam
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat




