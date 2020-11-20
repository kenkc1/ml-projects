""" CLASSIFICATION: This program uses the sklearn generated dataset make_moons and a 
hard vote classifier to choose train our model. It also compares the 'score' of
the different classifiers.
"""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


X, y = make_moons(n_samples = 500, noise = 0.30, random_state=42)
#print(X) # X(500,2)
#print(y) # y(500,1) target values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

"Define our different models"
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
                 estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], 
                 voting='hard')

"Train the model on voting_clf"
voting_clf.fit(X_train,y_train)
#print(voting_clf)

"Prediction"
predict = voting_clf.predict(X_test)
#print(predict)

"Evaluation"
print(classification_report(y_test, predict))
print(accuracy_score(y_test, predict)) # this gives % of correct predictions

"Compare score of the individual models"
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_predict))






