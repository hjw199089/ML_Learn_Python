import os

from numpy import *

import matplotlib.pyplot as plt

#导入训练数据集
from ML_Learn.com.ML.Regression.Regres_LWLR import lwlr

xArr, yArr = lwlr.loadDataSet(os.getcwd() + '/resource/ex0.txt')

xMat = mat(xArr)
yMat = mat(yArr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')

# yHat = lwlr.lwlrTest(xArr,xArr,yArr,1.0)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# ax.plot(xSort[:,1],yHat[srtInd])

# yHat = lwlr.lwlrTest(xArr,xArr,yArr,0.01)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:,0,:]
# ax.plot(xSort[:,1],yHat[srtInd])
# #
yHat = lwlr.lwlrTest(xArr,xArr,yArr,0.003)
srtInd = xMat[:,1].argsort(0)
xSort = xMat[srtInd][:,0,:]
ax.plot(xSort[:,1],yHat[srtInd])

plt.show()












