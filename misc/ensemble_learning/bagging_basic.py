""" Bagging or Boostrap Aggregating involves training multiple models from 
"boostrap samples", then aggregating those model to create a better model
(meta model).

Note that bagging can be used on any model (Logistic, SVM etc.)
When bagging is performed with Decision Trees, this is called a Random Forest
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.25, random_state=53)
# Note that the entire dataset can be used to train the bootstrap models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# BaggingClf takes the arguments (Model, no. of models, boostrap (True) pasting (false).
# n_jobs = -1 ultilises all cpu cores)
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=160, bootstrap=True, n_jobs=-1,
    oob_score=True)
bag_clf.fit(X_train, y_train)

# Pick X-test from our dataset (1st element)
#X_test = X_train[ : 1]
#print(X_train)
y_pred = bag_clf.predict(X_test)

"Evaluation"
print(bag_clf.oob_score_) # out-of-bag score

print(accuracy_score(y_test, y_pred))



