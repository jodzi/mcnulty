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
from sklearn.metrics import accuracy_score
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
#==============================================================================

#==============================================================================
# '''Challenge 3'''
# X_train = train[:,1:]
# y_train = train[:,0]
# X_test = test[:,1:]
# y_test = test[:,0]
# 
# #neigh = KNeighborsClassifier(n_neighbors=3)
# #neigh.fit(X_train,y_train)
# 
# scores = []
# for i in range(1,21):
#     neigh = KNeighborsClassifier(n_neighbors=i)
#     neigh.fit(X_train, y_train)
#     scores.append((i, neigh.score(X_test, y_test)))
#     
# print 'Best k = ' + str(max(scores, key=itemgetter(1))[0])
#==============================================================================

#==============================================================================
# '''Challenge 4'''
# model = LogisticRegression()
# 
# model.fit(X_train, y_train)
# 
# print model.score(X_test, y_test)
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
# y_pred_alldem = make_demo(X)
# y_pred_allrep = make_rep(X)
# all_dem = sklearn.metrics.accuracy_score(df['Class Name'], y_pred_alldem)
# all_rep = sklearn.metrics.accuracy_score(df['Class Name'], y_pred_allrep)
# print 'Prediting all democrat accuracy: {0:.2f}'.format(all_dem)
# print 'Prediting all republican accuracy: {0:.2f}'.format(all_rep)
#==============================================================================

'''Challenge 6'''
