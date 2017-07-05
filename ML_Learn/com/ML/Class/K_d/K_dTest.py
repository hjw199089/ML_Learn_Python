from ML_Learn.com.ML.Class.K_d import tree

#======构建数集测试
dataSet, labelsMat = tree.createDataSet()
print(dataSet)
# [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]


#=====数集香农熵测试
print(tree.calcShannonEnt(dataSet))
#0.9709505944546686

res = tree.splitDataSet(dataSet, 0, 1)

print(res)
# [[1, 'yes'], [1, 'yes'], [0, 'no']]

#===最好分类特征的选择测试
print(tree.chooseBestFeatureToSplit(dataSet))
#0