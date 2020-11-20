""" CLASSIFIER
"""
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# --- generates data from moons dataset. Moon is a tuple (X, Y)
moon = datasets.make_moons(n_samples=1000, noise=0.4)
#print(type(moon))

X = moon[0] # size (1000,2)
Y = moon[1] # size (1000,1)  either 0 or 1 target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4)

"Train model"
#change no. nodes and check classification report for best value
tree_class =  DecisionTreeClassifier(max_leaf_nodes=4) 
tree_class.fit(x_train, y_train)

# ---OR use GridSearchCV to find best parameters for DecisionTreeClassifier
# test max-leaf nodes from 1 to 100, min_samples in order to split node
params = {'max_leaf_nodes':list(range(2,100)), 
          'min_samples_split': list(range(2,5)),
          'criterion':['gini', 'entropy']
          }
gridsearch = GridSearchCV(DecisionTreeClassifier(random_state=42), params,verbose=1, cv=3)

clf = gridsearch.fit(x_train, y_train)
print(clf)
print(tree_class)

print(gridsearch.best_params_) # shows the hyperparemters found
print(gridsearch.best_score_)

"Predictions"
predict = tree_class.predict(x_test)
predict2 = gridsearch.predict(x_test)

"Evaluation"
print(classification_report(y_test, predict))
print(confusion_matrix(y_test,predict))
print(classification_report(y_test, predict2 ))
print(confusion_matrix(y_test,predict2))

"Plot decision tree"
plt.figure(1) # needed to show more than 1 figure (set param to any number)
plot_tree(clf.best_estimator_) # plot model found from hyperparamter tuning
 # uncomment to show plot

plt.figure(2)
plot_tree(tree_class)
plt.show()



