from numpy import *
a = [1,2,3]
b = [[2,2,3],[3,3,3],[1,2,3]]
labelMat = mat(b)
alphas =  mat(a).transpose()
# fXi = multiply(alphas,labelMat).T
# print(fXi)
# [[1 2 3]
#  [2 4 6]
#  [3 6 9]]

print(labelMat)

print('\n' , labelMat[0,:].T)
c = labelMat*labelMat[0,:].T
#
print('\n' ,c)


