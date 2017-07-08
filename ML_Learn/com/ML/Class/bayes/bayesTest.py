from ML_Learn.com.ML.Class.bayes import bayes

# #数据准备-文本转向量测试
# postingList,classVec = bayes.loadDataSet()
# postingList.sort()
# print(postingList)
#
# vocabList = bayes.createVocabList(postingList)
# vocabList.sort()
# print("vocabList: ", vocabList)
#
# print("postingList[0]: ", postingList[0])
# returnVec = bayes.setOfWord2Vec(vocabList,postingList[0])
# print(returnVec)
# # [['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'], ['stop', 'posting', 'stupid', 'worthless', 'garbage']]
# # vocabList:  ['I', 'ate', 'buying', 'cute', 'dalmation', 'dog', 'flea', 'food', 'garbage', 'has', 'help', 'him', 'how', 'is', 'licks', 'love', 'maybe', 'mr', 'my', 'not', 'park', 'please', 'posting', 'problems', 'quit', 'so', 'steak', 'stop', 'stupid', 'take', 'to', 'worthless']
# # postingList[0]:  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']
# # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]

#朴素贝叶斯分类器训练函数
postingList,classVec = bayes.loadDataSet()

vocabList = bayes.createVocabList(postingList)

print("index of stupid: ",vocabList.index("stupid"))

trainMat = []
for postinDoc in postingList:
    trainMat.append(bayes.setOfWord2Vec(vocabList,postinDoc))

p0V,p1V,pAb = bayes.trainNB0(trainMat,classVec)
print("pAb: ",pAb)
print("p0V: ",p0V)
print("p1V: ",p1V)


#测试贝叶斯分类函数
bayes.testingNB()
# ['love', 'my', 'dalmation'] classified as:  0
# ['stupid', 'garbage'] classified as:  1