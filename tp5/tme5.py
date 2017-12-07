import numpy as np
from sklearn import manifold, datasets
from matplotlib import pyplot as plt
from  time import  time
from mpl_toolkits.mplot3d import Axes3D

# X = np.loadtxt("mnist2500_X.txt")
# labels = np.loadtxt("mnist2500_labels.txt")

# TSNE 3 dimensions
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = vizualize.TSNE(n_components=3).fit_transform(X)
# plt.scatter(X[:,0],X[:,1], X[:,2], c = labels)
# plt.show()


# TSNE 2 dimensions
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = manifold.TSNE(n_components=2).fit_transform(X)
# plt.scatter(X[:,0],X[:,1], c = labels)
# plt.show()

# MDS 3 dimensions
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X = vizualize.MDS(n_components=3).fit_transform(X)
# plt.scatter(X[:,0],X[:,1], X[:,2], c = labels)
# plt.show()
#
#
# # MDS 2 dimensions
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='2d')
# X = vizualize.MDS(n_components=2).fit_transform(X)
# plt.scatter(X[:,0],X[:,1], c = labels)
# plt.show()


n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 3

#ISOMAP
# n_neighbors
# n_components

from matplotlib.ticker import NullFormatter

# fig = plt.figure(figsize=(15, 5))
#
# t0 = time()
# Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
# t1 = time()
# plt.title("Iso (%.2g sec)" % (t1 - t0))
# ax = fig.add_subplot(257)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
#
#
# t0 = time()
# mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# Y = mds.fit_transform(X)
# t1 = time()
# ax = fig.add_subplot(258)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("MDS (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
#
#
# t0 = time()
# se = manifold.SpectralEmbedding(n_components=n_components,
#                                 n_neighbors=n_neighbors)
# Y = se.fit_transform(X)
# t1 = time()
# ax = fig.add_subplot(259)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("SE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
#
#
# t0 = time()
# tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# Y = tsne.fit_transform(X)
# t1 = time()
# # print("t-SNE: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(2, 5, 10)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
#
#
# t0 = time()
# Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto').fit_transform(X)
# t1 = time()
# #print("%s: %.2g sec" % ("LocallyLinearEmbedding", t1 - t0))
# ax = fig.add_subplot(252 )
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("%s (%.2g sec)" % ("LLE", t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
#
# plt.show()



fig = plt.figure(figsize=(15, 5))

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
plt.title("Iso (%.2g sec)" % (t1 - t0))
ax = fig.add_subplot(257,  projection='3d')
plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2],  c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
ax = fig.add_subplot(258,  projection='3d')
plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2],  c=color, cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
ax = fig.add_subplot(259,  projection='3d')
plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2],  c=color, cmap=plt.cm.Spectral)
plt.title("SE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(2, 5, 10,  projection='3d')
plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2],  c=color, cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


t0 = time()
Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,eigen_solver='auto').fit_transform(X)
t1 = time()
ax = fig.add_subplot(252, projection='3d' )
plt.scatter(Y[:, 0], Y[:, 1],Y[:, 2], c=color, cmap=plt.cm.Spectral)
plt.title("%s (%.2g sec)" % ("LLE", t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()