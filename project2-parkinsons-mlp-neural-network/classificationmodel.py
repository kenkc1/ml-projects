''' Written by Kenny Cai
Project 2

This program explores the changes in tweaking hyperparameters for a
Keras sequential neural network. It allows users to pick and amount of
experimential runs and reports metrics, confidence intervals, a confusion 
matrix and an ROC curve.
'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


def reset_seed():
    ''' Function to reset seed for the reproducibility of results.
    '''
    seed_value = 42
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value) 
    tf.random.set_seed(seed_value)

def data_process():
    ''' Imports data, pre-processes it and returns the feature (X) and target data (y)
    '''
    # Import data and preprocessing
    data = pd.read_csv('classificationdata.csv', 
                names = ['Subject id', 'Jitter local', 'Jitter local absolute', 
                         'Jitter rap', 'Jitter ppq5', 'Jitter ddp', 
                         'Shimmer local', 'Shimmer local dB', 'Shimmer apq3', 
                         'Shimmer apq5', 'Shimmer apq11', 'Shimmer dda', 'AC', 
                         'NTH', 'HTN', 'Median pitch', 'Mean pitch', 
                         'Standard deviation', 'Minimum pitch', 'Maximum pitch',
                         'Number of pulses', 'Number of periods', 
                         'Mean period', 'Standard deviation of period', 
                         'Fraction of locally unvoiced frames', 
                         'Number of voice breaks', 'Degree of voice breaks',
                         'UPDRS', 'class information']
                         )

    # drops irrelevant columns (subject id, UPDRS)
    data = data.drop(['Subject id', 'UPDRS'], axis=1)
    
    # Select appropriate columns for X and y
    y = data.iloc[:, -1]
    X_raw = data.iloc[:, :-1]
    
    # Scale all feature columns between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X_raw)

    return X, y


def run_experiments(X_train, y_train, X_test, y_test):
    ''' This function uses a FOR loop to run through experiments of
    fitting a sequential model. It prints the mean metrics, then
    returns the metric arrays, and the fitted model (history).
    '''
    loss = []
    accuracy = []
    recall = []
    precision = []
    auc = []

    for i in range(1,2): # change numbers for required trials
        # reset_seed()

        # Declare the model. Change hyperparamters here for different models
        # Parameters here are given from the 'BEST MODEL'
        model = keras.models.Sequential([
            keras.layers.Dense(25, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        # Uncomment for different optimizer/ learning rates/ momentum
        #sgd = keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.99)
        adam = keras.optimizers.Adam(learning_rate=0.0005)

        model.compile(loss="binary_crossentropy", 
                      optimizer=adam, 
                      metrics=["accuracy", "Recall", "Precision", "AUC"]
                      )

        # Fit model
        history = model.fit(X_train, y_train.values, 
                            epochs=100, validation_split=0.4, 
                            verbose=0
                            )
        # Evaluate model on hold-out set (X_test)
        result = model.evaluate(X_test, y_test.values, verbose=0)

        # Appends list for metrics
        loss.append(result[0])
        accuracy.append(result[1])
        recall.append(result[2])
        precision.append(result[3])
        auc.append(result[4])

    # Rounds all metrics to 3 decimal places
    accuracym = round(np.asarray(accuracy).mean(), 3)
    lossm = round(np.asarray(loss).mean(), 3)
    recallm = round(np.asarray(recall).mean(), 3)
    precisionm = round(np.asarray(precision).mean(), 3)
    aucm = round(np.asarray(auc).mean(), 3)

    print('MEAN VALUES\nAccuracy = ', accuracym,'\nLoss = ', lossm, '\nRecall = ',recallm, '\nPrecision = ', precisionm, '\nAUC =', aucm)
        
    return accuracy, loss, recall, precision, auc, history, model


def confidence_intervals(ac, lo, re, pr, au):
    ''' Takes parameters as lists and calculates a 95% cconfidence interval
    '''
    accuracy = ac
    loss = lo
    recall = re
    precision = pr
    auc = au

    # Calculates 95% confidence intervals
    ci_accuracy = st.t.interval(0.95, len(accuracy)-1, loc=np.mean(accuracy), scale=st.sem(accuracy))
    ci_loss = st.t.interval(0.95, len(loss)-1, loc=np.mean(loss), scale=st.sem(loss))
    ci_recall = st.t.interval(0.95, len(recall)-1, loc=np.mean(recall), scale=st.sem(recall))
    ci_precision = st.t.interval(0.95, len(precision)-1, loc=np.mean(precision), scale=st.sem(precision))
    ci_auc = st.t.interval(0.95, len(auc)-1, loc=np.mean(auc), scale=st.sem(auc))

    # Rounds all the values
    ci_accuracy = tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, ci_accuracy))
    ci_loss = tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, ci_loss))
    ci_recall = tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, ci_recall))
    ci_precision = tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, ci_precision))
    ci_auc = tuple(map(lambda x: isinstance(x, float) and round(x, 3) or x, ci_auc))

    print('95% Confidence Intervals\nAccuracy = ', ci_accuracy, '\nLoss = ',
          ci_loss, '\nRecall = ', ci_recall, '\nPrecision = ',
          ci_precision, '\nAUC =', ci_auc
          )

def experiment_plot(kerasdict):
    ''' Plots a single experiment trial of the training and validation vs epochs
    '''
    history = kerasdict
    all_results = history.history
    compact_results = dict((k, all_results[k]) for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy'))
    pd.DataFrame(compact_results).plot(figsize=(8, 5))
    plt.xlabel(xlabel="Epochs")
    plt.ylabel(ylabel="Score")
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    plt.show()
    plt.close()

def confusion_matrix_plot(model):
    ''' Plots a confusion matrix for the model trained in the run_experiments function
    '''
    predict = model.predict_classes(X_test)
    matrix = confusion_matrix(y_test, predict) 
    class_names = ['Negative', 'Positive']
    dataframe_Confusion = pd.DataFrame(matrix, 
                                      index=class_names, 
                                       columns=class_names
                                       ) 

    sns.heatmap(dataframe_Confusion, annot=True,  cmap="Blues", fmt=".0f")
    plt.title("Confusion Matrix - best model")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('./confusion_matrix.png')
    plt.show()
    plt.close()

    return predict


#def classification_report_show(y_test, predict):
#    ''' Prints classfication report rearranged into a dataframe
#    '''
#    report = classification_report(y_test, predict, output_dict=True)
#    df_classification_report = pd.DataFrame(report).transpose()
#    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    
#    return df_classification_report


#def roc_auc(model, X_test, y_test):
    # Plots ROC curve and prints AUC score

#    predict_prob = model.predict_proba(X_test)
#    fpr, tpr, thresholds = roc_curve(y_test, predict_prob)

#    plt.plot(fpr, tpr)
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.0])
#    plt.title("ROC curve for PD classifier")
#    plt.xlabel("False Positive Rate (1-Specificity)")
#    plt.ylabel("True Postitive Rate (Sensitivity)")
#    plt.show()
#    plt.close()

#    auc_score = roc_auc_score(y_test, predict_prob)
#    print("AUC score = ", auc_score)
#'''

if __name__ == '__main__':
    X,y = data_process() # Processes data and returns feature and target as np arrays

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state = 42)

    accuracy, loss, recall, precision, auc, history, model = run_experiments(X_train, y_train, X_test, y_test) # run experiments

# ALL CODE BELOW IS COMMENTED OUT BECAUSE OF ED ISSUE
#    confidence_intervals(accuracy, loss, recall, precision, auc) # print confidence intervals
    
#    experiment_plot(history) # plots a single experiment of train and validation

#    predict = confusion_matrix_plot(model) 

#   print(classification_report_show(y_test, predict))

#    roc_auc(model, X_test, y_test)





