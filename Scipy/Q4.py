import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def func(x):
    return x**2 + 10*np.sin(x)


x0 = np.random.uniform(-10, 10)
res = optimize.fmin_bfgs(func, x0)
print("Optimization result using fmin_bfgs:", res)
x = np.linspace(-10, 10, 1000)
y = func(x)
plt.plot(x, y)
plt.plot(res, func(res), 'ro')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = x**2 + 10*sin(x)')
plt.show()

res = optimize.fminbound(func, -10, 10)
print("Optimization result using fminbound:", res)
x = np.linspace(-10, 10, 1000)
y = func(x)
plt.plot(x, y)
plt.plot(res, func(res), 'ro')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = x**2 + 10*sin(x)')
plt.show()

res = optimize.brute(func, ((-10, 10),))
print("Optimization result using brute:", res)
x = np.linspace(-10, 10, 1000)
y = func(x)
plt.plot(x, y)
plt.plot(res, func(res), 'ro')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = x**2 + 10*sin(x)')
plt.show()
