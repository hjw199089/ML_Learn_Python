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


'''
OLS
'''
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # 行列式的值是否为0，判断是否可逆
        print("矩阵不可逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws #返回归回系数


