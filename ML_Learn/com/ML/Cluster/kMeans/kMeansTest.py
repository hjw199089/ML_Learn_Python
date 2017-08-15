import os
from numpy import *
from ML_Learn.com.ML.Cluster.kMeans import kMeans
#导入训练数据集
inputMat = mat(kMeans.loadDataSet(os.getcwd() + '/resource/testSet.txt'))
centroids,clusterAssment = kMeans.binKmeans(inputMat,4)
print("centroids:\n",centroids)
print("clusterAssment:\n",clusterAssment)
kMeans.showCluster(inputMat, 4, centroids, clusterAssment)