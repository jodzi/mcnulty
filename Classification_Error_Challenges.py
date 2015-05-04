# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:59:18 2015

@author: josephdziados
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.learning_curve import learning_curve

#def change_vals(char):
#    if char == 'y':
#        return 1
#    elif char == 'n':
#        return 0
#
#func = [change_vals for i in range(18)]
#converter = dict(zip(range(1,18), func))
#
#columns = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime, duty-free-exports', 'export-administration-act-south-africa']
#
#votes = pd.read_csv('votes.csv', index_col=False, names=columns, converters=converter)
#
# Calculates the mean 
#means = [votes[columns[i]].mean() for i in range(1,len(votes.columns))]
## Creates the dictionary of means to replace NA's
#column_means_dict = dict(zip(votes.columns[1:], means))
#votes = votes.fillna(column_means_dict, inplace=True)
## Changes labels to 1's and 0's
#votes = votes.replace(to_replace=['democrat', 'republican'], value=[1, 0])

''' Challenge 1'''

#train, test = train_test_split(votes)
#
#X_train, y_train = train[:,1:], train[:,0]
#X_test, y_test = test[:,1:], test[:,0]
#
#scores = []
#
#for i in range(1,21):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn = knn.fit(X_train, y_train)
#    scores.append(knn.score(X_test, y_test))     
#    
#best_k = scores.index(max(scores)) + 1
#
## Create models
#knn_model = KNeighborsClassifier(n_neighbors=best_k)
#logistic_model = LogisticRegression()
#gnb_model = GaussianNB()
#svm_model = SVC(probability=True)
#dt_model = DecisionTreeClassifier()
#rf_model = RandomForestClassifier()
#
#models = [knn_model, logistic_model, gnb_model, svm_model, dt_model, rf_model]
#
#accuracy = []
#precision = []
#recall = []
#f_score = []
#probs = []
#  
#for model in models:
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    probs.append(model.predict_proba(X_test))
#    accuracy.append(accuracy_score(y_test, y_pred))
#    precision.append(precision_score(y_test, y_pred))
#    recall.append(recall_score(y_test, y_pred))
#    f_score.append(f1_score(y_test, y_pred))
#    
#print 'Knn n = {0}:'.format(best_k)
#print 'Accuracy Score: {0}'.format(accuracy[0])
#print 'Precision Score: {0}'.format(precision[0])
#print 'Recall Score: {0}'.format(recall[0])
#print 'F-score: {0}\n'.format(f_score[0])
#
#print 'Logistic Regression:'
#print 'Accuracy Score: {0}'.format(accuracy[1])
#print 'Precision Score: {0}'.format(precision[1])
#print 'Recall Score: {0}'.format(recall[1])
#print 'F-score: {0}\n'.format(f_score[1])
#
#print 'Gaussian Naive Bayes:'
#print 'Accuracy Score: {0}'.format(accuracy[2])
#print 'Precision Score: {0}'.format(precision[2])
#print 'Recall Score: {0}'.format(recall[2])
#print 'F-score: {0}\n'.format(f_score[2])
#
#print 'Support Vector Machines:'
#print 'Accuracy Score: {0}'.format(accuracy[3])
#print 'Precision Score: {0}'.format(precision[3])
#print 'Recall Score: {0}'.format(recall[3])
#print 'F-score: {0}\n'.format(f_score[3])
#
#print 'Decision Trees:'
#print 'Accuracy Score: {0}'.format(accuracy[4])
#print 'Precision Score: {0}'.format(precision[4])
#print 'Recall Score: {0}'.format(recall[4])
#print 'F-score: {0}\n'.format(f_score[4])
#
#print 'Random Forests:'
#print 'Accuracy Score: {0}'.format(accuracy[5])
#print 'Precision Score: {0}'.format(precision[5])
#print 'Recall Score: {0}'.format(recall[5])
#print 'F-score: {0}'.format(f_score[5])
#    
#'''Challenge 2'''
#roc = []
#auc_list = []
#
#for i, model in enumerate(models):
#    model.fit(X_train, y_train)
#    y_pred = model.predict(X_test)
#    probas_ = model.predict_proba(X_test)
#    fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
#    roc.append((fpr, tpr, thresholds))
#    auc_list.append(auc(fpr, tpr))
#    
#labels = ['KNN ' '(area = %0.2f)' %auc_list[0], 'Logistic ' '(area = %0.2f)' %auc_list[1], 'Naive Bayes ' '(area = %0.2f)' %auc_list[2], 'SVM ' '(area = %0.2f)' %auc_list[3], 'Decision Trees ' '(area = %0.2f)' %auc_list[4], 'Random Forest ' '(area = %0.2f)' %auc_list[5]]
#
#for i in range(5):
#    plt.plot(roc[i][0], roc[i][1], label=labels[i])
#plt.legend(loc='lower right') 
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curves for Various Classification Algorithms')

'''Challenge 3'''

#def calc_values(classifier, X, y, model_name):
#    
#    accuracy = np.mean(cross_val_score(classifier, X, y))
#    precision = np.mean(cross_val_score(classifier, X, y, scoring='precision'))
#    recall = np.mean(cross_val_score(classifier, X, y, scoring='recall'))
#    f_score = np.mean(cross_val_score(classifier, X, y, scoring='f1'))
#    
#    return model_name +  '\nAccuracy Score: {0}'.format(accuracy) + \
#                            '\nPrecision Score: {0}'.format(precision) + \
#                            '\nRecall Score: {0}'.format(recall) + \
#                            '\nF-score: {0}\n'.format(f_score)
#                            
#X = votes.values[:,1:]
#y = votes.values[:,0]
#
#print calc_values(knn_model, X, y, 'Knn = ' + str(best_k))
#print calc_values(logistic_model, X, y, 'Logistic Regression:')
#print calc_values(gnb_model, X, y, 'Naive Bayes:')
#print calc_values(svm_model, X, y, 'Support Vector Machines:')
#print calc_values(dt_model, X, y, 'Decision Trees:')
#print calc_values(rf_model, X, y, 'Random Forest:')


'''Challenge 4'''

'''Challenge 5'''
#hab_columns = ['age', 'operation_year', 'num_positive_nodes', 'survival']
#hab_df = pd.read_csv('haberman.csv', names=hab_columns)
#hab_df = hab_df.replace(to_replace = [1,2], value = [1,0])
#
#hab_train, hab_test = train_test_split(hab_df)
#
#X_hab_train, y_hab_train = hab_train[:,:3], hab_train[:,3] 
#X_hab_test, y_hab_test = hab_test[:,:3], hab_test[:,3]
#
#hab_roc = []
#hab_auc_list = []
#
#hab_log_model = LogisticRegression()
#hab_log_model.fit(X_hab_train,y_hab_train)
#hab_y_pred = hab_log_model.predict(X_hab_test)
#hab_probas_ = hab_log_model.predict_proba(X_hab_test)
#hab_fpr, hab_tpr, hab_thresholds = roc_curve(y_hab_test, hab_probas_[:,1])
#hab_roc = hab_fpr, hab_tpr, hab_thresholds
#hab_auc = auc(hab_fpr, hab_tpr)
#
#plt.plot(hab_roc[0], hab_roc[1], label='Logistic Regression' '(area = %0.2f)' %hab_auc)
#plt.legend(loc='lower right') 
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Logistic Regression on Haberman DataSet')