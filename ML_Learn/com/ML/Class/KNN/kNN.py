from notebook.notebookapp import raw_input
from numpy import *
import operator

from numpy.ma import array


def createDataSet():
    #创建数据集和标签
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
    return group,labels

#========K-紧邻分类器
# 用于分类的输入向量是inX,
# 输入的训练样本集为dataSet,
# 标签向量为labels,
# 最后的参数k表示用于选择最近邻居的数目,
# 其中标签向量的元素数目和矩阵dataSet的行数相同
# 使用欧氏距离公式
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # 计算完所有点之间的距离后, 可以对数据按照从小到大的次序排序
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 按照分类的频次排序为逆序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回发生频率最高的元素标签
    return sortedClassCount[0][0]

#=========从输入文件filename解析数据
#返回特征矩阵和分类标签
#文件数据样例：
#每年获得的飞行常客里程数 玩视频游戏所耗时间百分比  每周消费的冰淇淋公升数
#40920	8.326976	0.953952	largeDoses
#26052	1.441871	0.805124	didntLike
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    #初始化一个numberOfLines * 3 的零数组
    returnMat = zeros((numberOfLines,3))
    #标签列表
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        #获得特征
        returnMat[index,:] = listFromLine[0:3]
        #获得分类
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#======归一化处理
def autoNum(dataSet):
    minVals = dataSet.min(0) # 1*3
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet)) #同行列矩阵
    m = dataSet.shape[0] #行数
    # tile(matrix,(m,n)) 将matrix 沿x轴复制n次,y轴复制m次
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet,ranges,minVals

#======测试
#已给定样本集的随机的90%为训练样本，测试分类效果
def datingClassTest(filename,k,ration):
    hoRation = ration
    datingMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNum(datingMat)
    m = normMat.shape[0] # 总样本数，行数
    numTestVecs = int(m*hoRation) #测试样本数
    errorCount = 0.0 #错误分类数
    #遍历分类10%
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("分类错误率为: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
    print(numTestVecs)


#========分类器使用
# 􏵒得到个人并􏰚􏰈他的信息。 程􏰓会给出􏵆对喜欢程度
# filename 训练数据
def classifyPerson(filename):
    resultList = ['not at all','in small doses','in large doese']
    #个人参数
    # 每年获得的飞行常客里程数
    # 玩视频游戏所耗时间百分比
    # 每周消费的冰淇淋公升数
    persentTabs = float(raw_input("persentage of time spent playing video games?: "))
    ffMiles = float(raw_input("frequest filer miles cosumed per year?: "))
    iceCream = float(raw_input("liters of ice cream consumed per year?: "))
    #解析训练数据集
    datingDataMat,datingLabels = file2matrix(filename)
    #归一化数据
    normalMat, ranges, minVals = autoNum(datingDataMat)
    #个人特征值
    inArr = array([ffMiles,persentTabs,iceCream])
    #分类结果, 注意将输入特征值也要归一化
    classifierResult = classify0((inArr/minVals)/ranges,normalMat,datingLabels,3)
    print("You will probably like this person: ",resultList[classifierResult-1])

