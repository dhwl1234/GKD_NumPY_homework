import numpy as np
from scipy.sparse import dok_matrix, lil_matrix, coo_matrix

# 创建稀疏矩阵
sparse_dok = dok_matrix((4, 4), dtype=np.int32)
sparse_lil = lil_matrix((4, 4), dtype=np.int32)

# 设置矩阵元素
sparse_dok[0, 0] = 3
sparse_dok[0, 2] = 8
sparse_dok[1, 1] = 2
sparse_dok[3, 3] = 1

sparse_lil[0, 0] = 3
sparse_lil[0, 2] = 8
sparse_lil[1, 1] = 2
sparse_lil[3, 3] = 1

# 输出矩阵
print("DOK Matrix:\n", sparse_dok.toarray())
print("LIL Matrix:\n", sparse_lil.toarray())

# COO Matrix
data = np.array([3, 8, 2, 1])
row = np.array([0, 0, 1, 3])
col = np.array([0, 2, 1, 3])
sparse_coo = coo_matrix((data, (row, col)), shape=(4, 4))

# 输出矩阵
print("COO Matrix:\n", sparse_coo.toarray())
