"""
This is a Linear regression program to find the parameters for a
diabetes dataset
https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset
""" 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

"Load relavant dataset"
# --- get dataset from sklearn datbase
# --- diabetes data set
diabetes = datasets.load_diabetes()
#print(diabetes)

X = diabetes.data # --- feature values 442 rows x 10 features
#print(X.shape)
#print(diabetes.feature_names)
Y = diabetes.target # --- target values (y or dependant variable)
#print(Y)
#print(Y.shape)
#print(diabetes.target_names)  # 0 is setosa, 1 is versocolor, 3 is virginica

"Split the data into 80% training and 20% testing"
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

"Train the logistic regression model"
model = LinearRegression()
model.fit(x_train, y_train)

# --- shows all the parameter values for 10 features and the intercept at the end
print(model.coef_, model.intercept_)

"Test the model"
predict = model.predict(x_test)



"Model evaluation"
# --- R score
rscore = model.score(x_test,y_test)
print('R score is: ', rscore)

# --- mean squared error
error = np.mean((predict - y_test)**2)
print('MSE is: ', error)

