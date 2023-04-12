import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad

# def func(x):
#     return np.cos(np.exp(x))**2
# result, error = quad(func, 0, 3)
# print("The integral of cos^2(e^x) from 0 to 3 is: ", result)






def f(x, y):
    return 16 * x * y


def h(y):
    return (1-4 * y **2)**0.5

result, _ = dblquad(f, 0,0.5,lambda x:0,lambda x:h(x))
print("The double integral is:", result)
