import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# 生成数据点
x = np.array([-1, 0, 2.0, 1.0])
y = np.array([1.0, 0.3, -0.5, 0.8])

# 定义插值点的密集程度，并生成插值点
xi = np.linspace(-3, 4, 100)

# 使用Rbf函数进行插值
rbf_multiquadric = Rbf(x, y, function='multiquadric')
yi_multiquadric = rbf_multiquadric(xi)

rbf_gaussian = Rbf(x, y, function='gaussian')
yi_gaussian = rbf_gaussian(xi)

rbf_linear = Rbf(x, y, function='linear')
yi_linear = rbf_linear(xi)

# 绘制图形并比较三种插值方法的结果
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(xi, yi_multiquadric, label='Multiquadric Rbf')
plt.plot(xi, yi_gaussian, label='Gaussian Rbf')
plt.plot(xi, yi_linear, label='Linear Rbf')
plt.legend()
plt.show()
