# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:53:17 2015

@author: josephdziados
"""
import csv
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import sklearn.metrics
from sklearn.learning_curve import learning_curve
from operator import itemgetter

#==============================================================================
# '''Challenge 1'''
# with open('house-votes-84.data', 'r') as congress_file:
#     reader = csv.reader(congress_file)
#     data = open('house-votes-84.csv', 'w')
#     writer = csv.writer(data)
#     for row in reader:
#         writer.writerow(row)
#     data.close()
#     
# def change_vals(char):
#     if char == 'y':
#         return 1
#     elif char == 'n':
#         return 0
# 
# func = [change_vals for i in range(18)]
# converter = dict(zip(range(1,18), func))
# 
# columns = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime, duty-free-exports', 'export-administration-act-south-africa']
# 
# df = pd.read_csv('votes.csv', index_col=False, names=columns, converters=converter)
# 
# means = [df[columns[i]].mean() for i in range(1,len(df.columns))]
# column_means_dict = dict(zip(df.columns[1:], means))
# df = df.fillna(column_means_dict, inplace=True)
#==============================================================================
# Alternatively use df = df.fillna(df.mean())


#==============================================================================
# '''Challenge 2'''
# train, test = train_test_split(df)
# 
# '''Challenge 3'''
# X_train = train[:,1:]
# y_train = train[:,0]
# X_test = test[:,1:]
# y_test = test[:,0]
# 
# scores = []
# 
# for i in range(1,21):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn = knn.fit(X_train, y_train)
#     #y_pred = knn.predict(X_test)
#     #accuracy = np.where(y_pred==y_test, 1, 0).sum() / float(len(test))
#     #score = sklearn.metrics.accuracy_score(y_test, y_pred)
#     #scores.append(accuracy)
#     scores.append(knn.score(X_test, y_test))     
#     
# #print 'Best k = ' + str(max(scores, key=itemgetter(1))[0])
# best_k = scores.index(max(scores)) + 1
# print 'Best k = ' + str(best_k)
#==============================================================================

#==============================================================================
# '''Challenge 4'''
# logistic_model = LogisticRegression()
# logistic_model.fit(X_train, y_train)
# log_score = logistic_model.score(X_test, y_test)
# print log_score
#==============================================================================

#==============================================================================
# '''Challenge 5'''
# df['Class Name'].value_counts().plot(kind='bar', rot=0, title='Total Number of Republicans and Democrats')
# 
# print df['Class Name'].value_counts()
# 
# def make_demo(X):
#     return ['democrat'] * len(df)
# 
# def make_rep(X):
#     return ['republican'] * len(df)
# 
# X = df.values[:,1:]
# y_actual = df.values[:,0]
# y_pred_alldem = make_demo(X)
# y_pred_allrep = make_rep(X)
# all_dem = sklearn.metrics.accuracy_score(y_actual, y_pred_alldem)
# all_rep = sklearn.metrics.accuracy_score(y_actual, y_pred_allrep)
# print 'Predicting all democrat accuracy: {0:.2f}'.format(all_dem)
# print 'Predicting all republican accuracy: {0:.2f}'.format(all_rep)
#==============================================================================

#==============================================================================
# '''Challenge 6'''
# plt.plot(range(1,21), scores, 'r-', range(1,21), [log_score] * 20, 'b--', range(1,21), [all_dem] * 20, 'g-', range(1,21), [all_rep] * 20, 'c--')
# plt.xticks(np.arange(1,21))
#==============================================================================

#==============================================================================
# '''Challenge 7'''
# # Logistic Regression
# log_model = LogisticRegression()
# m, train_err, test_err = sklearn.learning_curve.learning_curve(log_model, X, y_actual)
# 
# train_cv_error = np.mean(train_err, axis=1)
# test_cv_error = np.mean(test_err, axis=1)
# 
# plt.figure()
# plt.plot(m, train_cv_error, label='Training') 
# plt.plot(m, test_cv_error, label='Test')
# plt.title('Logistic Regression Learning Curve')
# plt.legend(loc='best')
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy')
# 
# # K Nearest Neighbors
# knn_model = KNeighborsClassifier(n_neighbors=6)
# 
# knn_m, knn_train_err, knn_test_err = sklearn.learning_curve.learning_curve(knn_model, X, y_actual)
# 
# knn_train_cv_error = np.mean(knn_train_err, axis=1)
# knn_test_cv_error = np.mean(knn_test_err, axis=1)
# 
# plt.figure()
# plt.plot(knn_m, knn_train_cv_error, label='Training')
# plt.plot(knn_m, knn_test_cv_error, label='Test')
# plt.xlabel('Training Set Size')
# plt.ylabel('Accuracy')
# plt.title('KNN Learning Curve')
# plt.legend(loc='best')
#==============================================================================

#==============================================================================
# '''Challenge 8'''
# # Gaussian Naive Bayes
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# gnb_acc = gnb.score(X_test, y_test)
# print 'Gaussian Naive Bayes Accuracy: {0:.2f}'.format(gnb_acc)
# 
# # Support Vector Machine
# svm = SVC()
# svm.fit(X_train, y_train)
# svm_acc = svm.score(X_test, y_test)
# print 'SVM Accuracy: {0:.2f}'.format(svm_acc)
# 
# # Decision Tree
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# dt_acc = dt.score(X_test, y_test)
# print 'Decision Tree Accuracy: {0:.2f}'.format(dt_acc)
# 
# # Random Forest
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# rf_acc = rf.score(X_test, y_test)
# print 'Random Forest Accuracy: {0:.2f}'.format(rf_acc)
#==============================================================================

#==============================================================================
# '''Challenge 9'''
# # Gaussian Naive Bayes
# acc_gnb = np.mean(sklearn.cross_validation.cross_val_score(gnb, X, y_actual))
# print 'Gaussian Naive Bayes: {0:.2f}'.format(acc_gnb)
# # SVM
# acc_svm = np.mean(sklearn.cross_validation.cross_val_score(svm, X, y_actual))
# print 'SVM: {0:.2f}'.format(acc_svm)
# # Decision Tree
# acc_dt = np.mean(sklearn.cross_validation.cross_val_score(dt, X, y_actual))
# print 'Decision Tree: {0:.2f}'.format(acc_dt)
# # Random Forest
# acc_rf = np.mean(sklearn.cross_validation.cross_val_score(rf, X, y_actual))
# print 'Random Fores: {0:.2f}'.format(acc_rf)
#==============================================================================

#'''Challenge 10'''
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
#df2 = pd.read_csv('votes.csv', index_col=False, names=columns, converters=converter)
 
#df2 = df2.fillna(df2.mode())
#df2 = df2.replace(0.0, '0')
#df2 = df2.replace(1.0, '1')
 
#clf = KNeighborsClassifier()
#train_2, test_2 = train_test_split(df2)

#X_train_2 = train_2[:, 2:]
#y_train_2 = train_2[:,1]
#X_test_2 = test_2[:,2:]
#y_test_2 = test_2[:,1]

#clf.fit(X_train_2, y_train_2)
#print clf.score(X_test_2, y_test_2)