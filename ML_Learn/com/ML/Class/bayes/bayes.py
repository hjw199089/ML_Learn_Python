# 《机器学习实战》贝叶斯分类学习

from numpy import *


# 构建数据样本和集合标签集
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 标签 1 标识有侮辱性 0 否
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 数据集词条去重(利用set特性)
def createVocabList(dataSet):
    # 构建空集合
    vocabSet = set([])
    for document in dataSet:
        # 并集计算，将数据添加至vocabSet
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 将数据集转化为数据向量
# 根据词条是否在词表中出现过转化为向量
def setOfWord2Vec(vocabList, inputSet):
    # 返回的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 朴素贝叶斯分类器训练函数
# 文档矩阵，每篇文档标签的向量[1,0,xxxxxxx]
def trainNB0(trainMatrix, trainCategory):
    numTranDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # 总的词数目
    # 计算标签1出现的概率
    pAbusive = sum(trainCategory) / float(numTranDocs)
    # 解决个别为0的应用，初识化为1，分母初识化为2
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = 0.0; p1Denom = 0.0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0
    # 遍历所有文档
    for i in range(numTranDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 解决小数据下溢问题，取对数
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
# log(p(w|c)p(c)) = log(p(w|c)) + log(p(c))
# 对于分类都除p(w),比较大小时可以略去
def classifyNB(vec2Classify, p0Vec, p1vec, pClass1):
    p1 = sum(vec2Classify * p1vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 测试贝叶斯分类函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print (testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
