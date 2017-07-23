import os

from ML_Learn.com.ML.Class.SVM import svmMLiA

#导入训练数据集
dataArr, labelArr = svmMLiA.loadDataSet(os.getcwd() + '/resource/testSet.txt')

# #简单版SMO测试
# b,alphas = svmMLiA.smoSimple(dataArr,labelArr,0.6,0.001,40)
# print("b: " , b)
# print("alphas>0: ", alphas[alphas>0])
# for i in range(100):
#     if alphas[i]>0.0:
#         print("支持向量元素:\t" ,dataArr[i] ,"\t" ,labelArr[i] )
#
# # b:  [[-3.84148046]]
# # alphas>0:  [[ 0.14709994  0.17249089  0.04916758  0.00392681  0.36483161]]
# # 支持向量元素:	 [4.658191, 3.507396] 	 -1.0
# # 支持向量元素:	 [3.457096, -0.082216] 	 -1.0
# # 支持向量元素:	 [2.893743, -1.643468] 	 -1.0
# # 支持向量元素:	 [5.286862, -2.358286] 	 1.0
# # 支持向量元素:	 [6.080573, 0.418886] 	 1.0


b,alphas = svmMLiA.smoPK(dataArr,labelArr,0.6,0.001,40)
print("b: " , b)
print("alphas>0: ", alphas[alphas>0])