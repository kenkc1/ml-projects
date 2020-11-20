from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt


" Generates a quadratic training set + noise "
np.random.seed(42) # random start point for generating random numbers
m = 200

X1 = np.random.rand(m, 1) 
# generates random values between 0 and 1 in the shape 200 x 1
# for the x_train

y = 4 * (X1 - 0.5) ** 2 # y target values
y = y +np.random.randn(m, 1)/10 # adds some random noise
plt.scatter(X1, y)
plt.show()

"Build a regression tree with depth 2"

tree_reg = DecisionTreeRegressor(max_depth=2) # can change depth
tree_reg.fit(X1, y)

"Predict values"
x= 0.25 # change this (x_test)
y_predict = tree_reg.predict([[x]])
print(y_predict)