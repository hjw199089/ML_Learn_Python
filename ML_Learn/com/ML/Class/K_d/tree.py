from  math import log
import operator


# ========构建数集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# ======计算给􏴫数集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt


# ========􏾖􏳟􏲔􏲕􏾗􏾖􏳟􏲔􏲕􏾗划分数据集
# 􏷁􏺌分的数据􏱙、􏺌分数据􏱙的􏲯􏲰、需要返回的特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# 􏴀􏴪=====最好的数据􏱙划分特征值的选择
# 选取特征划分数据集，计算最好的划分数据集的特征
# 默认列表形式存储数据，最后一个元素为类别标签,其余为特征
# 先计算原始的信息熵，用于和划分之后比较
# 遍历计算用每一个特征划分数据的信息增益，取增益最大的特征为最好的数据􏱙划分特征值

def chooseBestFeatureToSplit(dataSet):
    #默认列表形式存储数据，最后一个元素为类别标签,其余为特征
    numFeatures = len(dataSet[0]) - 1
    # 先计算原始的信息熵，用于和划分之后比较
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain  = 0.0
    bestFeature  = -1
    #遍历每一个特征
    #[xx1,  xx2,  xx3,  label]
    for i in range(numFeatures):
        #第i个特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算新的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGian = baseEntropy - newEntropy
        if(infoGian > bestInfoGain):
            bestInfoGain = newEntropy
            bestFeature = i
    return bestFeature



