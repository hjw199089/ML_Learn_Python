import os
from numpy import *
from ML_Learn.com.ML.Regression.Regres_RegTree import regTrees

testMat    = mat(eye(4))
mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)       #
print (testMat)
print ("mat0:\n" ,mat0 )
print ("mat1:\n" , mat1)
#导入训练数据集
inputData = regTrees.loadDataSet(os.getcwd() + '/resource/ex00.txt')
inputMat = mat(inputData)
retTree = regTrees.createTree(inputMat)
print(retTree)
