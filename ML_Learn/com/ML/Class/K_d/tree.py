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

#计算熵
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


#划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
#数据最优划分方式
#计算最优特征进行数据集的划分
#计算原始信息熵，用于比较
#遍历每一个特征划分数据集的信息增益，取最大增益对应的特征划分数据集
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

#出现次数最多的分类名称
def majority(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(0),reverse = True)
    return sortedClassCount[0][0]

#构建决策树
#利用以上模块基于最好的属性划分数据集，每个数据集上再次最优划分，以次递归
#递归的结束条件，遍历完所有的数据集属性，或者每个分支下的所有实例都有相同的分类;
#或者特征使用完仍然不能唯一分类数据，采用多数表决的方式选择该数据块的分类标签
def createTree(dataSet,labels):
    #类别
    classList = [example[-1] for  example in dataSet]
    #数据集中所有的类别都一样了，结束递归
    if classList.count(classList[0]) == len(classList):
        return classList[0] #返回类别
    if len(dataSet[0]) == 1: #没有特征可用了，结束递归
        return  majority(classList) #对本组内的类别进行多数表决
    #选择最优特征进行数据划分（分支）
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    #用一个map存储决策树，其中值为label代表节点，为map代表分支
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        suLables = labels[:] #List 作为参数传递时 是指针型，会被修改，这里拷贝一份
        #返回的树作为当前节点的子节点
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),suLables)
    return myTree

#测试决策树
def classify(inputTree,featureLables,testVec):
    firstStr = list(inputTree.keys())[0]#根节点
    secondDict = inputTree[firstStr]
    #根节点在 输入向量中的位置
    featIndex = featureLables.index(firstStr)
    key = testVec[featIndex]
    #得到改特征的分类数据块
    valueOfFeat = secondDict[key]
    #判断是否为叶子节点了,否则继续递归
    if isinstance(valueOfFeat,dict):
        classLabel = classify(valueOfFeat,featureLables,testVec)
    else: classLabel = valueOfFeat
    return classLabel





