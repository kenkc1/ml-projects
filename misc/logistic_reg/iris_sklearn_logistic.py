"""
This is a Logistic regression program to classify iris species
https://www.youtube.com/watch?v=ACdBKML9l4s
""" 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

"Load relavant dataset"
# --- get dataset from sklearn datbase
# --- iris data set
iris = datasets.load_iris()


X = iris.data # --- feature values 150 x 4 features
#print(iris.feature_names)
Y = iris.target # --- target values (y or dependant variable)
#print(Y)
#print(iris.target_names)   0 is setosa, 1 is versocolor, 3 is virginica

""" Visualise the data
plt.xlabel('Features')
plt.ylabel('Species')

pltX = X[:,0]
pltY = Y
plt.scatter(pltX, pltY, color='blue', label='sepal_length')

pltX = X[:,1]
pltY = Y
plt.scatter(pltX, pltY, color='red', label='sepal_width')

pltX = X[:,2]
pltY = Y
plt.scatter(pltX, pltY, color='green', label='petal_length')

pltX = X[:,3]
pltY = Y
plt.scatter(pltX, pltY, color='black', label='petal_width')

plt.legend()
plt.show()
"""

"Split the data into 80% training and 20% testing"
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

"Train the logistic regression model"
model = LogisticRegression()
model.fit(x_train, y_train)

"Test the model"
predict = model.predict(x_test)
print('Predictions are:\n',predict)
print('Target values are:\n', y_test)

"Model evaluation"
# classification report (check precision, recall, f1-score)
print(classification_report(y_test, predict))
print(confusion_matrix(y_test,predict))
