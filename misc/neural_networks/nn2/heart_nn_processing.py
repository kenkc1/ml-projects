import pandas as pd 
import numpy as np

data = pd.read_csv('heart.csv')

X_raw = data.iloc[:,:13]
y_raw = data.iloc[:,13]

y_raw = y_raw.to_frame()   # need this line for onehotencoder
#print(type(y_raw))

# scaler [0,1] all features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X_raw)  # numpy array

#print(type(X_scaled))

# one-hot encode target
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)   
y_encoded = ohe.fit_transform(y_raw) # numpy array

#print(type(y_encoded))
#print(y_encoded)

# Combine the 2 numpy arrays back into table

data_new = np.concatenate((X_scaled, y_encoded), axis=1)
#print(data_new)

np.savetxt('processed_data.csv', data_new, delimiter=',')

# 15mins


