'''Project 1 Part A - Interpretable model - Decision Tree Written by Kenny Cai '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.compose import make_column_transformer

data = pd.read_csv('./data.csv')
data = data.iloc[:, 1:] # removes first column with id's
X_raw = data.iloc[:,:-1] # creates feature matrix without churn

''' Data pre-processing'''
# One hot encoding
encoder = OneHotEncoder(sparse = False)
column_trans = make_column_transformer(
    (encoder, ['gender','SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
         'Contract', 'PaperlessBilling', 'PaymentMethod']), 
    remainder = 'passthrough')
#encoder.fit_transform(X_raw[['PaymentMethod']])
#encoder.categories_
X = column_trans.fit_transform(X_raw)

# Binary encode churn
target = data.iloc[:,-1:]
y = target.apply(LabelEncoder().fit_transform)

'''Train test split'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

''' Hyperprameter tuning '''
params = {'max_depth' : list(range(1,11))}
#'min_samples_leaf' : [75,80,85,90,100,110,120],
#'min_samples_split': [1,2,3,4,5,8]}
gridsearch = GridSearchCV(DecisionTreeClassifier(random_state=6), 
                          params,verbose=1, cv=5, 
                          scoring = 'roc_auc')

# Fit model and print best parameters found
clf = gridsearch.fit(X_train, y_train)
best_param = clf.best_params_
print('Thes best params are', best_param, '\n')

'''BEST MODEL'''
best_model_task1 = DecisionTreeClassifier(max_depth=5, random_state=42)
best_model_task1.fit(X_train, y_train)

'''Model Evaluation'''
# Check overfitting by predicting the training set. Single iteration.
predict_train = best_model_task1.predict(X_train) # predict for training set
acc = accuracy_score(y_train, predict_train)
pre = precision_score(y_train, predict_train)
rec = recall_score(y_train, predict_train)
print('TRAIN SET SINGLE prediction scores:\nAccuracy score =', acc)
print('Precision score =', pre)
print('Recall score =', rec, '\n')

# Scores for single iteration of test data
predict = best_model_task1.predict(X_test) 
acc = accuracy_score(y_test, predict)
pre = precision_score(y_test, predict)
rec = recall_score(y_test, predict)
print('TEST SET SINGLE:\nAccuracy score =', acc)
print('Precision score =', pre)
print('Recall score =', rec, '\n')

# 10 fold Cross validation predict scores on hold out dataset 
#a = clf.best_estimator_

y_predict = cross_val_predict(best_model_task1, X_test,y_test, cv =10)
acc = accuracy_score(y_test, y_predict)
pre = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
print('TEST SET 10 FOLD CV (hold out data set)\nAccuracy score =', acc)
print('Precision score =', pre)
print('Recall score =', rec)

#scores = cross_val_score(a, X_test, y_test, cv=10, scoring = 'recall') # Shows the cross val scores for each metric

'''Plot the confusion Matrix'''
matrix = confusion_matrix(y_test, y_predict)
class_names = ['Churn_No', 'Churn_Yes']
dataframe_Confusion =  pd.DataFrame(matrix, index=class_names, columns=class_names) 

sns.heatmap(dataframe_Confusion, annot=True,  cmap="Blues", fmt=".0f")
plt.title("Confusion Matrix - DT")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.savefig('./confusion_matrix.png')
plt.show()
plt.close()

'''Plot the Decision Tree'''
#plot_tree(a)
#plt.show()
#plt.close()