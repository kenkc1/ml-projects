# --- https://www.youtube.com/watch?v=b0L47BeklTE
# --- OLS NOT GRADIENT DESCENT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# gets data from cvs and points them into numpy array (100,2) shape
points = np.genfromtxt('data_linearreg.csv', delimiter=',')

x=points[:,0]
y=points[:,1]

# --- graphs the points
#plt.scatter(x,y)
#plt.show()

# --- does a train, test, split
# --- x_train and y_train contains 75/100 data points
x_train, x_test, y_train, y_test = train_test_split(x,y)

# --- illustrates the testing and training data split
#plt.scatter(x_train,y_train, color ='r')
#plt.scatter(x_test, y_test, color = 'g')
#plt.show()

# --- function to train only x_test and y_test
LR = LinearRegression()
# --- Linear regression function when using entire dataset to train model
LR2 = LinearRegression()

# --- Fit regression model, the reshape is needed for it to work
LR.fit(x_train.reshape(-1,1),y_train)

LR2.fit(x.reshape(-1,1),y)

# --- prediction is the equation of the line
prediction = LR.predict(x_test.reshape(-1,1))

# --- plot the testing points with the line of best fit.
#plt.plot(x_test,prediction, color = 'r')
#plt.scatter(x_train,y_train, color = 'b')
#plt.show()

# --- prints coefficients of linear model
print(LR2.coef_)
print(LR2.intercept_)