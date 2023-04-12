import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def spring_system(state, t, M, k, b, F):
    x, x_dot = state
    x_dot_dot = (F - b * x_dot - k * x) / M
    return [x_dot, x_dot_dot]

M, k, b, F = 1.0, 0.5, 0.2, 1.0
init_state = [-1.0, 0.0]
t = np.arange(0, 50, 0.02)
solution = odeint(spring_system, init_state, t, args=(M, k, b, F))

plt.plot(t, solution[:, 0])
plt.xlabel('time (ms)')
plt.ylabel('x')
plt.show()
