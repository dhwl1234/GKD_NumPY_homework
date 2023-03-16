import numpy as np
import time
import math


x = [i * 0.001 for i in range(1000000)]
start = time.clock()
for i, t in enumerate(x):
    x[i] = math.sin(t)
print("math.sin:", time.clock() - start)
