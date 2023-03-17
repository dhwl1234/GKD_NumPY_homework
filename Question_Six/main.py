import numpy as np

# 定义系数矩阵A和常数向量b
A = np.array([[3, 6, -5], [1, -3, 2], [5, -1, 4]])
b = np.array([12, -2, 10])

# 解方程
x = np.linalg.solve(A, b)

print(x)
