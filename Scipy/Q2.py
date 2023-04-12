import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 生成真实的高斯分布数据
x = np.linspace(0, 10, 101)
a_true, b_true, c_true = 1, 5, 2
y_true = a_true * np.exp(-(x - b_true)**2 / (2 * c_true**2))

# 加入噪声生成实际数据
np.random.seed(0)
y_noise = y_true + 0.1 * np.random.randn(len(y_true))

# 定义高斯分布的函数
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# 使用curve_fit函数拟合
popt, pcov = curve_fit(gaussian, x, y_noise)
a_fit, b_fit, c_fit = popt
y_fit = gaussian(x, a_fit, b_fit, c_fit)

# 绘制图形并比较拟合结果和真实数据
plt.plot(x, y_true, label='True Data')
plt.scatter(x, y_noise, label='Noisy Data')
plt.plot(x, y_fit, label='Fitted Data')
plt.legend()
plt.show()
