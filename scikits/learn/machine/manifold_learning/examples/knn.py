import numpy as np
import matplotlib.pyplot as plt
from scikits.learn.machine.manifold_learning.regression.neighbors import \
     Neighbors, Kneighbors, Parzen


n = 100 # number of points
data1 = np.random.randn(n,2) + 3.0
data2 = np.random.randn(n, 2) + 5.0
data = np.concatenate((data1, data2))
labels = [0]*n + [1]*n

# we create the mesh
h = 0.1 # step size
x = np.arange(-2, 12, h)
y = np.arange(-2, 12, h)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)


neigh = Neighbors(data, labels=labels, k=6)

for i in range(len(x)):
    for j in range(len(y)):
        Z[i,j] = neigh.predict((x[i], y[j]))

ax = plt.subplot(111)
plt.pcolormesh(X, Y, Z)
plt.scatter(data1[:,0], data1[:,1])
plt.scatter(data2[:,0], data2[:,1], c='red')

plt.show()
