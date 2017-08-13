from numpy import *


def loadDataSet(fileName):
    dataMat = [];
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 转为浮点数存储
        dataMat.append(fltLine)
    return dataMat


# 根据特征及其取值二分数据集
# dataSet = array ([[1,0,2,0],[0,0,2,0],[1,0,2,0]])
# a= nonzero(dataSet[:,0] > 0.5)
# print(a)
# #(array([0, 2]),)
# a= nonzero(dataSet[:,0] > 0.5)[0]
# print(a)
# # [0 2]
def binSplitDataSet(dataSet, feature, value):
    val1 = nonzero(dataSet[:, feature] > value)[0]
    val2 = nonzero(dataSet[:, feature] <= value)[0]

    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1


# 叶子节点的值（采用均值）
def regLeaf(dataSet):
    return mean(dataSet[:, -1])


# 误差函数（总方差）
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 最佳切分点查询函数
# 返回切分点特质值及其最佳取值
# 这里设置一个停止条件ops（tols和tolN）
# 分别代表容许的误差下降值和切分样本的最小样本数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0];
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)  # 初识化误差
    bestS = inf;
    bestIndex = 0;
    bestValue = 0  # 不断用最小误差更新
    for featureIndex in range(n - 1):  # 遍历个特征
        for splitVaule in set((dataSet[:, featureIndex].T.A.tolist())[0]):  # 遍历各个特征的各个值
            mat0, mat1 = binSplitDataSet(dataSet, featureIndex, splitVaule)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featureIndex
                bestValue = splitVaule
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# createTree()
# 找到最佳的待切分特征:
#     若该节点不能再切分，将该节点作为叶子返回
#     执行二元切分
#     在右子树调用createTree()
#     在左子树调用createTree()
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val  # 切分完毕
    # 构建树
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
