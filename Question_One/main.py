import numpy as np
import time


def triangle_wave(x, c, c0, hc):
    x = x - x.astype(int)  # 三角波的周期为1，因此只取x坐标的小数部分进行计算
    return np.where(x >= c, 0, np.where(x < c0, x / c0 * hc, (c - x) / (c - c0) * hc))


def triangle_wave2(x, c, c0, hc):
    x = x - x.astype(int)
    return np.select([x >= c, x < c0, True], [0, x / c0 * hc, (c - x) / (c - c0) * hc])


def triangle_wave3(x, c, c0, hc):
    x = x - x.astype(int)
    return np.piecewise(x,
                        [x >= c, x < c0],
                        [0,  # x>=c
                         lambda x: x / c0 * hc,  # x<c0
                         lambda x: (c - x) / (c - c0) * hc])  # else


def triangle_wave4(x, c, c0, hc):
    """显示每个分段函数计算的数据点数"""

    def f1(x):
        return x / c0 * hc

    def f2(x):
        return (c - x) / (c - c0) * hc

    x = x - x.astype(int)
    return np.piecewise(x, [x >= c, x < c0], [0, f1, f2])


x = np.linspace(0, 2, 1000)

start = time.perf_counter()
y = triangle_wave(x, 0.6, 0.4, 1.0)
print("y:time of using where nest ,  :", time.perf_counter() - start)

start = time.perf_counter()
y2 = triangle_wave2(x, 0.6, 0.4, 1.0)
print("y2:time of using select ,  :", time.perf_counter() - start)

start = time.perf_counter()
y3 = triangle_wave3(x, 0.6, 0.4, 1.0)
print("y3:time of using piecewise :", time.perf_counter() - start)

start = time.perf_counter()
y4 = triangle_wave4(x, 0.6, 0.4, 1.0)
print("y4:time of using traditional selection struction :", time.perf_counter() - start)

if np.alltrue(y == y2) and np.alltrue(y2 == y3) and np.alltrue(y3 == y4):
    print("The four ways of Generating triangular wave have the same result")
