import numpy as np
import matplotlib.pyplot as plt
# gets data from cvs and points them into numpy array (100,2) shape
points = np.genfromtxt('data_linearreg.csv', delimiter=',')

# scatter plot of points
#plt.scatter(points[:,0],points[:,1])
#plt.show()

# print first 20 points
#print(points[:-80])

learning_rate = 0.0001
n_iters = 1000
initial_b = 0
initial_m = 0

N = float(len(points))

m_current = 0
b_current = 0


for i in range(0,n_iters):
    # to sum the derivatives for each iteration, begin each at derivative at 0
    b_gradient = 0
    m_gradient = 0

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += (2/N)*((m_current * x + b_current)-y)
        m_gradient += (2/N)*((m_current * x + b_current)-y)*x

    b_current = b_current - (learning_rate * b_gradient)
    m_current = m_current - (learning_rate * m_gradient)

print(b_current,m_current)

