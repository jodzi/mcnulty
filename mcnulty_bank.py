# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:47:25 2015

@author: josephdziados
"""

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit, \
                                        StratifiedKFold, permutation_test_score
from sklearn.feature_selection import RFECV, SelectPercentile, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, accuracy_score, precision_score, \
                            recall_score, f1_score, auc, roc_curve, roc_auc_score, classification_report
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler, scale
import pickle
import seaborn as sns

def model_search(estimator, tuned_params, scores, X_train, y_train, X_test, y_test):  
    
    cv = ShuffleSplit(len(X_train), n_iter=3, test_size=0.30, random_state=0)

    for score in scores:
        print"# Tuning hyper-parameters for %s" % score
        print

        clf = GridSearchCV(estimator, tuned_params, cv=cv,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)

        print"Best parameters set found on development set:"
        print
        print clf.best_params_
        print
        print "Grid scores on development set:"
        print
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
        print

        print "Detailed classification report:"
        print
        print "The model is trained on the full development set."
        print "The scores are computed on the full evaluation set."
        print
        y_true, y_pred = y_test, clf.predict(X_test)
        print classification_report(y_true, y_pred)
        print

bank = pd.read_csv('bank-additional-full.csv', sep=';') 
# Replace yes/no with 1 and 0
bank = bank.replace(['yes','no'], [1,0])
# Puts the outcome variable at the front of the dataframe
cols = bank.columns.tolist()
cols = cols[-1:] + cols[:-1]
bank = bank[cols]
# Drops the unknowns from the data
bank.replace(['unknown'], [np.nan], inplace=True)
bank = bank.dropna()
# Delete features that can't be used for new clients
del bank['duration']
del bank['campaign']
del bank['pdays']
del bank['previous']
del bank['poutcome']
 
# Add dummies for categorical variables
bank = pd.get_dummies(bank, columns=['month', 'day_of_week', 'job', 'marital', 'education', 'contact'])
 
# Delete a column for each categorical variable to use as a base class
del bank['month_mar']
del bank['day_of_week_wed']
del bank['job_blue-collar']
del bank['marital_married']
del bank['education_university.degree']
del bank['contact_cellular']
 
# Randomly shuffle data
bank = bank.iloc[np.random.permutation(len(bank))]
 
# Scale continuous features
X = bank.values[:,1:].astype(float)
y = bank.values[:,0].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)
 
X_add_all = scale(X[:,(0,4,5,6,7,8)])
X_scaled = np.delete(X, (0,4,5,6,7,8), axis=1)
X_scaled = np.insert(X_scaled, (0,3,3,3,3,3), X_add_all, axis=1)
 
std_scale = StandardScaler(copy=True).fit(X_train[:,(0,4,5,6,7,8)])
X_add_train = std_scale.transform(X_train[:,(0,4,5,6,7,8)]) 
X_add_test = std_scale.transform(X_test[:,(0,4,5,6,7,8)])
X_train_scaled = np.delete(X_train, [0,4,5,6,7,8], axis=1)
X_train_scaled = np.insert(X_train_scaled, (0,3,3,3,3,3), X_add_train, axis=1)
X_test_scaled = np.delete(X_test, [0,4,5,6,7,8], axis=1)
X_test_scaled = np.insert(X_test_scaled, (0,3,3,3,3,3), X_add_test, axis=1)
 
# Model Selection
#tuned_params = [{'C': [.00001, .0001, .001, .01, .10, 1, 10, 100, 1000]}]
#scores = ['accuracy', 'precision', 'recall', 'roc_auc']
# 
#model_search(LogisticRegression(), tuned_params, scores, X_train_scaled, y_train, X_test_scaled, y_test)


logistic_model = LogisticRegression().fit(X_train_scaled, y_train)

with open('logistic_model','w') as model:
    pickle.dump(logistic_model, model)
    
with open('std_scale', 'w') as scaler:
    pickle.dump(std_scale, scaler)
