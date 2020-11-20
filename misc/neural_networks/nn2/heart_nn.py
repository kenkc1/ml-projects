import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, roc_auc_score, roc_curve
import os

os.environ['PYTHONSEED'] = str(42)
np.random.seed(42)
tf.random.set_seed(42)



data = pd.read_csv('processed_data.csv', header=None)

X = data.iloc[:, :13]
y = data.iloc[:, -1]

X = X.to_numpy()
y = y.to_numpy()
#print(y)

# create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#print(type(X_train))
#print(type(y_train))

# create ADAM neural network -------------------------------------------------------

model = keras.models.Sequential([
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0)

# plot training -----------------------------------------
'''
all_results = history.history
compact_results = dict((k, all_results[k]) for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy'))

pd.DataFrame(compact_results).plot()
plt.show()
'''

# Evaluate ADAM model ----------------------------------------
score = model.evaluate(X_test, y_test)
y_pred = model.predict_classes(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('RMSE: ', mean_squared_error(y_test, y_pred))

# create ADAM neural network -------------------------------------------------------
model2 = keras.models.Sequential([
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0)

'''
# plot training sgd
all_results2 = history2.history
compact_results2 = dict((k, all_results2[k]) for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy'))

pd.DataFrame(compact_results2).plot()
plt.show()
'''
# Evaluate SGD model ----------------------------------------
score2 = model2.evaluate(X_test, y_test)
y_pred2 = model2.predict_classes(X_test)

print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print('RMSE: ', mean_squared_error(y_test, y_pred2))


# ROC curve
adam_fpr, adam_tpr, _ = roc_curve(y_test, model.predict_proba(X_test))
sgd_fpr, sgd_tpr, _ = roc_curve(y_test, model2.predict_proba(X_test))

plt.plot(adam_fpr, adam_tpr, marker='.', label='adam')
plt.plot(sgd_fpr, sgd_tpr, marker='.', label='sgd')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_curve')
plt.show()

'''
adam optimizer - 200 epochs, 2 hidden layer (20 nerons)
loss: 0.4279 - accuracy: 0.8279
              precision    recall  f1-score   support

         0.0       0.78      0.83      0.80        52
         1.0       0.87      0.83      0.85        70

    accuracy                           0.83       122
   macro avg       0.82      0.83      0.83       122
weighted avg       0.83      0.83      0.83       122

[[43  9]
 [12 58]]
RMSE:  0.1721311475409836
sgd optimizer - 200 epochs, 2 hidden layer (20 nerons)
Results:
loss: 0.4851 - accuracy: 0.8197
              precision    recall  f1-score   support

         0.0       0.79      0.79      0.79        52
         1.0       0.84      0.84      0.84        70

    accuracy                           0.82       122
   macro avg       0.82      0.82      0.82       122
weighted avg       0.82      0.82      0.82       122

[[41 11]
 [11 59]]
RMSE:  0.18032786885245902

- Both optimizer's had very similar results.
- Adam had 12 false negatives and 9 false positive vs sgd that had 11 false negatives and 11 false postives.
- sgd classfied more positive predictions with 59 as oppsed to Adam with 58.
- training over epochs where also quite similar. Adam was not much faster for this dataset
- Adam optimizer performs better. Surprisingly the sgd optimizer was not too far off.

52 mins
'''