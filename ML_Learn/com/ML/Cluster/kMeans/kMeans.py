from numpy import *
import time
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 转为浮点数存储
        dataMat.append(fltLine)
    return dataMat


# 计算两个点间的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 随机质点生成
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 列数，也即特征的个数
    centroids = mat(zeros((k, n)))  # k行，n维
    for j in range(n):  # 对每一特征维度 求随机数
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 待分簇的数据个数
    clusterAssment = mat(zeros((m, 2)))  # m个数据点的分类簇结果
    # 其中第一列是簇编号，第二列是SSE
    # 产生随机k质点
    centroids = createCent(dataSet, k)
    # 直到所有点不能再改变簇
    clusterChanged = True
    cnt = 0
    while clusterChanged:
        clusterChanged = False
        cnt = cnt + 1
        print(cnt)
        # 对于每个点
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 对所有质点计算距离,求最小距离和所在簇
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 判断本次循环，改点是否所属簇改变
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 更新簇信息(不分区是否改变)
            clusterAssment[i, :] = minIndex, minDist ** 2
        for centItr in range(k):  # recalculate centroids
            ptsClust = dataSet[nonzero(clusterAssment[:, 0].A == centItr)[0]]
            centroids[centItr, :] = mean(ptsClust, axis=0)  # 对列（本特质维度）求均值
    return centroids, clusterAssment


# 二分K-Means
def binKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    #初始化首个质点和SSE
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # 取出第j个簇的所有数据,准备切分
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 求切分后的总的SSE
            sseSplit = sum(splitClustAss[:, 1])
            # 求其他为切分簇的SSE和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            if (sseSplit + sseNotSplit) < lowestSSE:# 切分后总的SSE 减小了
                bestCentToSplit = i # 最佳切分簇
                bestNewCents = centroidMat# 切分后的簇质点，序号 0 和 1
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新切分后的簇和数据的信息
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)   #将1号作为新增出来的质点
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 将0号作为切分前的质点
        # 同理更新质点
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        # 更新数据的SSE
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()
