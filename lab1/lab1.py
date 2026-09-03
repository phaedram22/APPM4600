import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2 * np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)
plt.plot(X, Ya)
plt.plot(X, Yb)

X = np.linspace(0, 2 * np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)
plt.plot(X, Ya)
plt.plot(X, Yb)
plt.xlabel("x")

plt.ylabel("x")
plt.show()

#3.2 exercises
x = np.linspace(0, 10, num=5)
y = np.arange(5)

print(x)
print(y)

print(y[2])
print("the first three entries of x are" f"{x[0:3]}")

w = 10**(-np.linspace(1,10,10))
print(w)
print(len(w))
x = np.arange(1,11)
print(x)
print(len(x))

plt.semilogy(x,w)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

s = 3*w
plt.semilogy(s, w)
plt.show()

