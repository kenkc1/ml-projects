import numpy as np
import matplotlib.pyplot as plt
# gets data from cvs and points them into numpy array (100,2) shape
points = np.genfromtxt('data_linearreg.csv', delimiter=',')

# scatter plot of points
#plt.scatter(points[:,0],points[:,1])
#plt.show()

# print first 20 points
#print(points[:-80])

#X=points[:, 0]
#y=points[:, 1]

X1 = points[:, np.newaxis, 0]
y = points[:, np.newaxis, 1]
#X = np.reshape(X, (len(points),1))

X_bias  = np.ones((len(X1),1))

X  = np.append(X_bias, X1, axis = 1)
XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)

XTy = np.dot(X.T, y)
theta = np.dot(XTX_inv, XTy)
print(theta)

#plt.scatter(X1,y)
#plt.show()
#print(np.linalg.det(XTX))
#print(np.dot(XTX_inv, XTX))

