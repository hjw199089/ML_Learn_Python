import os

from numpy import *
from ML_Learn.com.ML.Regression.BasicRegressionOLS import regression
import matplotlib.pyplot as plt

#导入训练数据集
xArr, yArr = regression.loadDataSet(os.getcwd() + '/resource/ex0.txt')
ws = regression.standRegres(xArr,yArr)
print("xArr[0:2]: \n", xArr[0:2])
print("yArr[0:2]: \n", yArr[0:2])
print("ws: \n" ,ws)
print("xArr[0:2]*ws.T = \n",mat(xArr[0:2])*ws)


xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat*ws
#判断模型的好坏-相关系数
cor = corrcoef(yHat.T,yMat)
print("相关系数=  \n", cor)
#
# xArr[0:2]:
#  [[1.0, 0.067732], [1.0, 0.42781]]
# yArr[0:2]:
#  [3.176513, 3.816464]
# ws:
#  [[ 3.00774324]
#  [ 1.69532264]]
# xArr[0:2]*ws.T =
#  [[ 3.12257084]
#  [ 3.73301922]]
# 相关系数=
#  [[ 1.          0.98647356]
#  [ 0.98647356  1.        ]]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()




