import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数g(x)
def g(x):
    z = (x - 1) * 5
    return np.sin(z**2) + np.sin(z)**2

# 指定插值区间和节点数
a, b = -1, 1
n = 100

# 在切比雪夫节点上插值
cheb_nodes = np.polynomial.chebyshev.chebpts1(n, (a, b))
cheb_fit = np.polynomial.chebyshev.Chebyshev.fit(cheb_nodes, g(cheb_nodes), n)
cheb_error = np.abs(g(np.linspace(a, b, 1000)) - cheb_fit(np.linspace(a, b, 1000))).max()

# 在等距节点上插值
poly_nodes = np.linspace(a, b, n)
poly_fit = np.polynomial.Polynomial.fit(poly_nodes, g(poly_nodes), n-1)
poly_error = np.abs(g(np.linspace(a, b, 1000)) - poly_fit(np.linspace(a, b, 1000))).max()

# 绘制插值结果和误差图像
x = np.linspace(a, b, 1000)
plt.plot(x, g(x), label='g(x)')
plt.plot(x, cheb_fit(x), label='Chebyshev fit')
plt.plot(x, poly_fit(x), label='Polynomial fit')
plt.legend()

plt.figure()
plt.bar(['Chebyshev', 'Polynomial'], [cheb_error, poly_error])
plt.ylabel('Maximum error')
plt.show()
