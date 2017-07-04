from ML_Learn.com.ML.Class.KNN import kNN

#========数据解析-测试集和标签测试=======
#输入文件样例：
#每年获得的飞行常客里程数 玩视频游戏所耗时间百分比  每周消费的冰淇淋公升数  喜好权重
# 40920	8.326976	0.953952	3
# 14488	7.153469	1.673904	2
filename = '/Users/hjw/Documents/Java/python/ML/com/ML/Class/KNN/resources/datingTestSet2.txt'
returnMat,classLabelVector = kNN.file2matrix(filename)
#查看训练集样例
print("训练集样例: \n",returnMat[0:2,:])
#查看标签
print("标签样例: \n",classLabelVector[0:2])

# 训练集样例:
#  [[  4.09200000e+04   8.32697600e+00   9.53952000e-01]
#  [  1.44880000e+04   7.15346900e+00   1.67390400e+00]]
# 标签样例:
#  [3, 2]

#=======分类器测试=========
print("======测试分类错误率: \n")
#50%的作为训练集，50%数据􏴊􏰑􏴓分测试􏰣器
kNN.datingClassTest(filename, 3, 0.5)
# 分类错误率为: 0.064000
# 32.0
# 500
#70%的作为训练集，30%数据􏴊􏰑􏴓分测试􏰣器
kNN.datingClassTest(filename, 3, 0.3)
# 分类错误率为: 0.080000
# 24.0
# 300
#90%的作为训练集，10%数据􏴊􏰑􏴓分测试􏰣器
kNN.datingClassTest(filename, 3, 0.1)
# 分类错误率为: 0.050000
# 5.0
# 100

#=====分类器应用
kNN.classifyPerson(filename)

# 分类错误率为: 0.050000
# 5.0
# 100
# persentage of time spent playing video games?: 10
# frequest filer miles cosumed per year?: 10000
# liters of ice cream consumed per year?: 0.5
# You will probably like this person:  in large doese


