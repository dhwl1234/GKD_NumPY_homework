import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 生成1000个从伽马分布中抽取的随机数
shape = 1
scale = 1
samples = gamma.rvs(a=shape, scale=scale, size=1000)

# 绘制直方图
# 其中，bins 表示直方图的箱子数量，density 表示是否将直方图规范化为概率密度函数（PDF）
# ，alpha 表示直方图的透明度，label 表示图例中显示的标签。
plt.hist(samples, bins=30, density=True, alpha=0.5, label='histogram')

# 绘制概率密度函数
x = np.linspace(0, 10, 100)
pdf = gamma.pdf(x, a=shape, scale=scale)
plt.plot(x, pdf, 'r-', label='gamma pdf')

# 添加图例并显示图像
plt.legend()
plt.show()
