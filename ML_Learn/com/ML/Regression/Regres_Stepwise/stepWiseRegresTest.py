import os

from numpy import *
import matplotlib.pyplot as plt


#导入训练数据集
from ML_Learn.com.ML.Regression.Regres_Stepwise import stepWiseRegres

xArr, yArr = stepWiseRegres.loadDataSet(os.getcwd() + '/resource/abalone.txt')

ridegeWeights = stepWiseRegres.stepWise(xArr, yArr,0.01,200)


