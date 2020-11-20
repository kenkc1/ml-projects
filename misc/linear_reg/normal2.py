# https://medium.com/@dikshitkathuria1803/normal-equation-using-python-5993454fbb41
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
x =  np.array([1,2,3,4,5]) #Uncomment this when using Sample  Dataset
y =  np.array([7,9,12,15,16])   #Uncomment this when using Sample Dataset
# dataset = pd.read_csv('Salary_Data.csv') #Uncomment this when using SalaryVsExp Dataset
# x = dataset.iloc[:, 0].values  #Uncomment this when using SalaryVsExp Dataset
# y = dataset.iloc[:, 1].values  #Uncomment this when using SalaryVsExp Dataset
plt.scatter(x,y,color='red')
x_bias = np.ones((5,1))
x = np.reshape(x,(5,1))
x = np.append(x_bias,x,axis=1)
#x_transpose = np.transpose(x)
#x_transpose_dot_x = x_transpose.dot(x)
#temp_1 = np.linalg.inv(x_transpose_dot_x)
#temp_2=x_transpose.dot(y)
#theta =temp_1.dot(temp_2)
#print(theta)
# y = 4.6 + 2.4*x            #Uncomment this when using Sample Dataset
# y = 25792.2 +  9449.96*x  #Uncomment this when using SalaryVsExp Dataset
#plt.plot(x,y,color='blue')
#plt.show()

print(x)
