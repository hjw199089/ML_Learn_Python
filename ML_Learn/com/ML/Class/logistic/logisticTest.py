from ML_Learn.com.ML.Class.logistic import logistic

dataMat,labelMat = logistic.loadDataSet('/Users/hjw/Documents/Java/python/ML_Learn/ML/ML_Learn/com/ML/Class/logistic/resources/testSet.txt')
weights = logistic.gradAscent(dataMat,labelMat)
print(weights)
# [[ 4.12414349]
#  [ 0.48007329]
#  [-0.6168482 ]]
