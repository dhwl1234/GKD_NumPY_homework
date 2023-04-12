from scipy.optimize import fsolve
import numpy as np

# 定义方程组的函数
def equations(p):
    a, r = p
    d = 140
    L = 156
    eq1 = np.cos(a) - 1 + d**2 / (2*r**2)
    eq2 = a*r - L
    return [eq1, eq2]

# 提供初始解
a, r = 1.0, 1.0
# 使用fsolve函数求解方程组
result = fsolve(equations, [a, r], fprime=None)

print(result)
