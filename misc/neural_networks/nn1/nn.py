import pandas as pd 
import numpy as np 

data = pd.read_csv('processed_data.csv', header=None)
#print(data.shape)   # 683 x 11

X = data.iloc[:, 0:9]
y = data.iloc[:, 10] # only need last row as it is a binary classification

#print(type(y))     # may need to change to_frame - acutally don't
#print(type(X))

X = X.to_numpy()
y = y.to_numpy()

# create train test split -----------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# create neural network -------------------------------
# CHANGE HERE FOR SGD VS. ADAM to train for each model
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], random_state=42)
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)

# evaluate neural network on test set --------------------------------
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_auc_score

score = model.evaluate(X_test, y_test)
print('Accuracy on test set: ', score[1])
print('Loss on test set: ', score[0])

# Predict classes
y_pred = model.predict_classes(X_test)
print(confusion_matrix(y_test ,y_pred))

AUC = roc_auc_score(y_test, y_pred)
print('AUC score: ', AUC)

rmse  = mean_squared_error(y_test, y_pred)
print('Mean squared error: ', rmse)

# Second model with SGD----------------------------------------------------
model2 = keras.models.Sequential([
    keras.layers.Dense(5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'], random_state=42)
history2 = model2.fit(X_train, y_train, epochs=20, validation_split=0.2, verbose=0)

# Evaluate Model 2 ------------------------------------------------------------
score2 = model2.evaluate(X_test, y_test)
print('Accuracy on test set: ', score2[1])
print('Loss on test set: ', score2[0])

# Predict classes
y_pred2 = model2.predict_classes(X_test)
print(confusion_matrix(y_test ,y_pred2))

AUC = roc_auc_score(y_test, y_pred2)
print('AUC score: ', AUC)

rmse2  = mean_squared_error(y_test, y_pred2)
print('Mean squared error: ', rmse2)

# Plot ROC and AUC curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_pred_prob = model.predict_proba(X_test)
y_pred_prob2 = model2.predict_proba(X_test)

print(type(y_pred_prob))
print(type(y_test))

'''
adam_fpr, adam_tpr, thresholds = roc_curve(y_test, y_pred_prob)
sgd_fpr, sgd_tpr, thresholds = roc_curve(y_test, y_pred_prob2)

plt.plot(adam_fpr, adam_tpr, linestyle='--', label='adam', )
plt.plot(sgd_fpr, sgd_tpr, marker='.', label='sgd', alpha=0.5)
plt.ylabel('True postive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()

'''