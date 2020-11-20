''' Project 1 Part B - Written by Kenny Cai
Exploring different ensembling models for a customer churn dataset.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('./data.csv')
# Data processing
data = data.iloc[:, 1:] # removes first column with id's
X_raw = data.iloc[:,:-1] # creates feature matrix without churn
#X_raw = X_raw.drop(['PhoneService','MultipleLines', 'OnlineBackup','DeviceProtection','StreamingTV','StreamingMovies'], axis = 1)

''' Data pre-processing'''
encoder = OneHotEncoder(sparse = False)
column_trans = make_column_transformer((encoder, ['gender','SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']), remainder = 'passthrough')
#encoder.fit_transform(X_raw[['PaymentMethod']])
#encoder.categories_
# NEW feature matrix
X = column_trans.fit_transform(X_raw)

# Binary encode churn
target = data.iloc[:,-1:]
y = target.apply(LabelEncoder().fit_transform)
''' Train test split '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

''' Random Forest Model searching '''
#'''------------------------------------------------------------------
# Hyperprameter tuning
#params = {'n_estimators' :[180, 200, 215, 230],
#         'max_depth': [8],
#         'min_samples_leaf': [18]}

#params = {'n_estimators' :[190,200,230],
 #        'max_depth': [7,8,9],
 #       'min_samples_split': [15,20,25]}

#random_grid = {
 #'max_depth': [6,8,10],
 #'max_features': ['auto', 'sqrt'],
 #'min_samples_leaf': [40, 50, 60],

 #'n_estimators': [ 300, 350,400]}
#gridsearch = GridSearchCV(RandomForestClassifier(max_features = 'sqrt', random_state=5, oob_score=True), params,verbose=1, cv=5, n_jobs=-1, scoring = 'roc_auc')
#gridsearch = RandomizedSearchCV(estimator = RandomForestClassifier(random_state=5, oob_score=True), param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring = 'roc_auc')
gridsearch = RandomForestClassifier(random_state=5, oob_score=True, n_estimators=215, min_samples_leaf=18,max_depth=8,max_features='sqrt')
#---------------------------------------------------------------'''

'''--BEST MODEL-------'''
best_model_task2 = RandomForestClassifier(random_state=5, oob_score=True, n_estimators=215, min_samples_leaf=18,max_depth=8,max_features='sqrt')
best_model_task2.fit(X_train,y_train.values.ravel())
'''------------------'''
clf = gridsearch.fit(X_train,y_train.values.ravel())

'''AdaBoost Model search'''
'''-----------------------------------------------------------
# Hyperprameter tuning
#
#params = {"base_estimator__max_depth" : [2,4,6,8,10],
 #             "n_estimators": [40, 50, 70, 90]
             #}
DTC = DecisionTreeClassifier()
ABC = AdaBoostClassifier(base_estimator = DTC)
random_grid = {'base_estimator__max_depth': [1,2,3,4,5,7], 
               'base_estimator__min_samples_leaf': [30,100,150,200,300], 
               'n_estimators': [50,80,100,150,200], 
               'learning_rate': [0.05, 0.1, 0.5, 1.0]}
#grid_search_ABC = GridSearchCV(ABC, params, n_jobs=-1, verbose=1)

gridsearch = RandomizedSearchCV(estimator = ABC, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring = 'roc_auc')
-----------------------------------------------------------------'''

'''Voting Classifier'''
'''--------------------------------------------------------------
DTC = DecisionTreeClassifier(random_state=5
ABC = AdaBoostClassifier(base_estimator = DTC)
RFC = RandomForestClassifier(random_state=5

clf = VotingClassifier(
    estimators=[('rf', RFC), ('abc', ABC)],
    voting='soft')

#params = {'rf__max_depth': [7,9,11,13,15,17],

#      'abc__n_estimators': [60, 80,100,140,180]}

random_grid = {
 'abc__base_estimator__max_depth': [1,3,5,7,9],
 'abc__base_estimator__min_samples_leaf': [150,180,200,220],
 'abc__n_estimators': [40,60, 80,100,120],
 'abc__learning_rate': [0.2,0.3, 0.05, 0.1],
 'rf__max_depth': [2,4,6,8,10],
 'rf__max_features':['auto', 'sqrt'],
 'rf__n_estimators': [150,200,220, 240],
 'rf__min_samples_leaf': [15,18,25,30]}

gridsearch = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring = 'roc_auc')
-----------------------------------------------------------'''

'''Evaluate Model'''
# Check overfitting by predicting the training set. Single iteration.
predict_train = clf.predict(X_train) # predict for training set
acc = accuracy_score(y_train, predict_train)
pre = precision_score(y_train, predict_train)
rec = recall_score(y_train, predict_train)
print('TRAIN SET SINGLE prediction scores:\nAccuracy score =', acc)
print('Precision score =', pre)
print('Recall score =', rec,'\n')

# Scores for single iteration of test data
predict = clf.predict(X_test) 
acc = accuracy_score(y_test, predict)
pre = precision_score(y_test, predict)
rec = recall_score(y_test, predict)
print('TEST SET SINGLE:\nAccuracy score =', acc)
print('Precision score =', pre)
print('Recall score =', rec,'\n')

#a = clf.best_estimator_
y_predict = cross_val_predict(clf, X_test,y_test.values.ravel(), cv =10)
acc = accuracy_score(y_test, y_predict)
pre = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
print('TEST SET 10 FOLD CV (hold out data set)\nAccuracy score =', acc)
print('Precision score =', pre)
print('Recall score =', rec)

matrix = confusion_matrix(y_test, y_predict)
# Plot confusion Matrix
class_names = ['Churn_No', 'Churn_Yes']
dataframe_Confusion =  pd.DataFrame(matrix, index=class_names, columns=class_names) 

sns.heatmap(dataframe_Confusion, annot=True,  cmap="Blues", fmt=".0f")
plt.title("Confusion Matrix Random Forest - Hold-out set")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.savefig('./confusion_matrix.png')
plt.show()
plt.close()
