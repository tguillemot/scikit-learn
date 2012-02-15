import pylab as pl
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

e = np.exp(1)

np.random.seed(4)


y = np.random.normal(scale=0.5, size=(30000))
x = np.random.normal(scale=0.5, size=(30000))
z = np.random.normal(scale=0.1, size=(len(x)), )
def pdf(x):
    return 0.5*(  stats.norm(scale=0.25/e).pdf(x)
                + stats.norm(scale=4/e).pdf(x))
density = pdf(x) * pdf(y)
pdf_z = pdf(5*z)

density *= pdf_z

a = x+y
b = 2*y
c = a-b+z

norm = np.sqrt(a.var() + b.var())
a /= norm
b /= norm

fig = pl.figure(1, figsize=(4, 3))
pl.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=-42, azim=-153)

pl.set_cmap(pl.cm.hot_r)

pts = ax.scatter(a[::10], b[::10], c[::10], c=density,
        marker='+', alpha=.4)
#mlab.outline(extent=[-3*a.std(), 3*a.std(), -3*b.std(), 3*b.std(),
#                     -3*c.std(), 3*c.std()])

Y = np.c_[a, b, c]
U, pca_score, V = np.linalg.svd(Y, full_matrices=False)
x_pca_axis, y_pca_axis, z_pca_axis = V.T*pca_score/pca_score.min()

#mlab.view(-20.8, 83, 9, [0.18, 0.2, -0.24])
#mlab.savefig('3d_data.jpg')
ax.quiver(0.1*x_pca_axis, 0.1*y_pca_axis, 0.1*z_pca_axis,
                x_pca_axis, y_pca_axis, z_pca_axis,
                color=(0.6, 0, 0))

x_pca_axis, y_pca_axis, z_pca_axis = 3*V.T
x_pca_plane = np.r_[x_pca_axis[:2], - x_pca_axis[1::-1]]
y_pca_plane = np.r_[y_pca_axis[:2], - y_pca_axis[1::-1]]
z_pca_plane = np.r_[z_pca_axis[:2], - z_pca_axis[1::-1]]

x_pca_plane.shape = (2, 2)
y_pca_plane.shape = (2, 2)
z_pca_plane.shape = (2, 2)

#mlab.mesh(x_pca_plane, y_pca_plane, z_pca_plane, color=(0.6, 0, 0),
#            opacity=0.1)
#mlab.mesh(x_pca_plane, y_pca_plane, z_pca_plane, color=(0.6, 0, 0),
#            representation='wireframe', line_width=1, opacity=0.3)

#mlab.view(-20.8, 83, 9, [0.18, 0.2, -0.24])
#mlab.savefig('3d_data_pca_axis.jpg')

# A view
#mlab.view(3.3, 43.8, 9.2, [0.04, -0.11, -0.17])
ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())
#pl.savefig('pca_3d.png')

