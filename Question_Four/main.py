import numpy as np

Z = np.random.random((5, 5))
Z_min, Z_max = Z.min(), Z.max()
Z_norm = (Z - Z_min) / (Z_max - Z_min)

print(Z_norm)
