from numpy import *
a = [1,2,3]
b = [[2,2,3],[3,3,3],[1,2,3]]
bMat = mat(b)
print(bMat)
print(mean(bMat,0))
# [[2 2 3]
#  [3 3 3]
#  [1 2 3]]
# [[ 2.          2.33333333  3.        ]]

print(bMat/mat([1,2,3]))

