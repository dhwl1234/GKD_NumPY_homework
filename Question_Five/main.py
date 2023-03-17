import numpy as np

# 定义数组
Z = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

# 给定值
z = 5.1

# 将数组展平成一维数组
Z_flat = Z.flatten()

# 计算每个元素与给定值之间的差值的绝对值
diff = np.abs(Z_flat - z)

# 找到差值的最小值及其索引
min_diff_index = np.argmin(diff)
min_diff_value = diff[min_diff_index]

# 返回原数组中对应索引的元素
result = Z_flat[min_diff_index]

print(result) # 输出：5
