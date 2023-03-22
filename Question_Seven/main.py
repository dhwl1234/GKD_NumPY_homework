import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Chebyshev

def g(x):
    z = (x - 1) * 5
    return np.sin(z**2) + np.sin(z)**2


n = 101
# Generate node
x = np.linspace(-1,1,n)
cheb_nodes = Chebyshev.basis(n).roots()
poly_nodes = np.polynomial.Polynomial.basis(n).roots()
xd = np.linspace(-1,1,1000)

# Interpolation calculate
cheb_fit = Chebyshev.fit(cheb_nodes,g(cheb_nodes),n-1,domain=[-1,1])
#poly_fit = np.polynomial.Polynomial.fit(x,g(x),n-1,domain=[-1,1])
poly_fit1= np.polynomial.Polynomial.fit(poly_nodes,g(poly_nodes),n-1,domain=[-1,1])
# Calculate error
cheb_error = np.abs(g(xd) - cheb_fit(xd)).max()
poly_error = np.abs(g(xd)-poly_fit1(xd)).max()

print(u"插值多项式的最大误差："),
print(u"ploynomial点：", poly_error),
print(u"chebyshev：",cheb_error)
# Draw picture
plt.plot(xd, g(xd), label='g(x)')
plt.plot(xd, cheb_fit(xd), label='Chebyshev fit')
plt.plot(xd, poly_fit1(xd), label='Polynomial fit')
plt.legend()

plt.figure()
plt.bar(['Chebyshev', 'Polynomial'], [cheb_error, poly_error])
plt.ylabel('Maximum error')
plt.show()
