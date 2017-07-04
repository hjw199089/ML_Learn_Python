from ML_Learn.com.ML.Class.KNN import kNN

#========测试集和标签测试
group,labels = kNN.createDataSet()
print("测试集: ",group)
print("分类标签: ",labels)

#========标签测试
classOutPut = kNN.classify0([1, 1.12], group, labels, 3)
print("分类输出: ",classOutPut)

# 测试集:  [[ 1.   1.1]
#  [ 1.   1. ]
#  [ 0.   0. ]
#  [ 0.   0.1]]
# 分类标签:  ['A', 'A', 'B', 'B']
# [('B', 2), ('A', 1)]
# B