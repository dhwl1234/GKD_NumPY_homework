import numpy as np

arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((len(arr)-1)*2, dtype=int)
result = np.insert(zeros, np.arange(0, len(arr))*2, arr)
print(result)
