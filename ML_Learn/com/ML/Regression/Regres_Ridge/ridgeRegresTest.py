import os

from numpy import *
import matplotlib.pyplot as plt


#导入训练数据集
from ML_Learn.com.ML.Regression.Regres_Ridge import ridgeRegres

xArr, yArr = ridgeRegres.loadDataSet(os.getcwd() + '/resource/abalone.txt')

ridegeWeights = ridgeRegres.ridgeRegresTest(xArr, yArr )

xMat = mat(xArr);
yMat = mat(yArr).T
yMean = mean(yMat, 0)
yMat = yMat - yMean  # 减去均值
xMeans = mean(xMat, 0)  # 计算每一维度下的均值
xVar = var(xMat, 0)  # 计算每一维度下的方差
xMat = (xMat - xMeans) / xVar
print("yArr = " ,yMat[0])
print("yArrHat = ",xMat[0,:]*mat(ridegeWeights[24,:]).T)

