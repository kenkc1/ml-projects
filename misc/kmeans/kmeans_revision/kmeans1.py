import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv('dataset_clustering.csv', header=None)

x = X.iloc[:, 0]
y = X.iloc[:, 1]

plt.scatter(x,y)
plt.show()
# use silhouette scores to pick optimum number of clusters ---------
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = []
inertias = []
for k in range(2,9):
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    score = silhouette_score(X, kmeans.labels_)
    inertia = kmeans.inertia_
    inertias.append(inertia)
    scores.append(score)

print(inertias)
print(scores)

# plot inertias and scores
plt.plot(range(2,9),inertias)
plt.savefig('inertias')
plt.show()

plt.plot(range(2,9), scores)
plt.savefig('scores')
plt.show()
plt.close()

'''
print(' silhouette_scores: \n', scores)
score_max = np.max(scores)
k_max = np.argmax(scores)  + 2
print('k_max: ', k_max, '\nscore_max: ' , score_max)

#cluster = KMeans(n_clusters=4)
#cluster.fit_predict(X)

'''
