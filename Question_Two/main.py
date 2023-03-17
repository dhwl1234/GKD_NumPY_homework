import numpy as np

arr11 = 5 - np.arange(1, 13).reshape(4, 3)

print("The original array formula is ",arr11)
# 1 Compute the sum of all elements and each column
print("Sum of all the elements", np.sum(arr11))
print("Sum of each column", np.sum(arr11, axis=0))
# 2 Compute the cumulative sum of each element and each column
print("Cumulative sum of all elements:", np.cumsum(arr11))
print("Cumulative sum of each column:", np.cumsum(arr11, axis=0))
# 3 Compute the cumulative product of each row
print("Cumulative product of each row:", np.cumprod(arr11, axis=1))
# 4 Compute the minimum value of all element
print("Minimum value of all elements:", np.min(arr11))
# 5 Compute the maximum value of each column
print("Maximum value of each column:", np.max(arr11, axis=0))
# 6 Compute the mean of all elements and each row
print("Mean of all elements:", np.mean(arr11))
print("Mean of each row:", np.mean(arr11, axis=1))
# 7 Compute the median of all elements and each column
print("Median of all elements:", np.median(arr11))
print("Median of each column:", np.median(arr11, axis=0))
# 8 Compute the variance of all elements and the standard deviation of each row
print("Variance of all elements:", np.var(arr11))
print("Standard deviation of each row:", np.std(arr11, axis=1))
