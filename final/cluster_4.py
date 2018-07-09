import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

Axes3D

with open('seq_4mer.csv', 'rb') as f:
    reader = csv.reader(f)
    kmers = list(reader)

for seq in kmers:

    for f in seq:

        seq[seq.index(f)] = int(f)

#print kmers[0]

X = np.asarray(kmers)

print "Fitting"
#kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

print X.shape

X_reduced = PCA(n_components = 6).fit_transform(X)

print X_reduced.shape

np.save('4mers_PCA_6.npy', X_reduced)

'''
X_embedded = TSNE(n_components=3, verbose = 1, n_iter = 250, early_exaggeration = 300).fit_transform(X_reduced)

print X_embedded.shape

print X_embedded[:3]

np.save('embedded_8mers_1.npy', X_embedded)
'''
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X_embedded[:,0], X_embedded[:,1],X_embedded[:,2])#,cmap = plt.cm.rainbow)
#plt.show()
