import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1) # dict with data, taget, categories etc.

data = mnist.data

X = mnist.data
y = mnist.target
X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]

# Train random forest classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

clf = RandomForestClassifier(random_state=42, verbose=1)
clf.fit(X_train, y_train)

# Get accuracy score for model
from sklearn.metrics import accuracy_score
predict = clf.predict(X_test)
acc = accuracy_score(y_test, predict)
print("Accuracy without PCA:", acc)

# Apply PCA and train again with reduced dataset
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95) # 95% variance
X_reduced = pca.fit_transform(X)
#print(X_reduced.shape)

X_train_red, y_train_red, X_test_red, y_test_red = X_reduced[:60000], y[:60000], X_reduced[60000:], y[60000:]

#Fit model with reduced dataset
clf2 = RandomForestClassifier(random_state=42, verbose=1)
clf2.fit(X_train_red, y_train_red)

predict2 = clf2.predict(X_test_red)
acc2 = accuracy_score(y_test_red, predict2)
print("accuracy with PCA:", acc2)