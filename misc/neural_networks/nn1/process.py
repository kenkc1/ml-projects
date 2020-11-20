import numpy as np
import pandas as pd

data = pd.read_csv('cancerdataraw.csv', header=None)

#print(type(data))                   # Get shape and type of data
#print(data.shape)
data = data.drop([0], axis=1) # drop index 0 of patient IDs

data = data[(data != '?').all(axis=1)] # drops all rows with ? in a row

X_raw = data.iloc[:, 0: 9]
y_raw = data.iloc[:, 9]
#print(y_raw.shape)



# scale all feature from 0 to 1 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X_raw)


# one-hot-encode target data
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
print(type(y_raw))     # need to change y_raw into a pd dataframe as fit_transform doesn't work for pd series
y_raw = y_raw.to_frame()  
print(type(y_raw))         
y = ohe.fit_transform(y_raw)
print(type(y))

# Note the encoder makes returns a numpy array, not the input dataframe

# combine the two processed dataframes X and y


processed_data = np.concatenate((X,y), axis=1)
#print(processed_data)

np.savetxt('processed_data.csv', processed_data,  delimiter=",")


# time taken - 35 mins