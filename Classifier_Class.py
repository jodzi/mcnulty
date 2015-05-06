# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:57 2015

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

bank = pd.read_csv('bank.csv', sep=';') 
bank = bank.replace(to_replace=['yes','no'], value=[1,0])


class Classifier(object):
    """
    builds a classifier object using an estimator model, an X array of features
    and a y response array.
    model = estimator
    X = numpy array
    y = numpy array
    """
    
    def __init__(self, model, dataframe, xcols, ycols): # X, y):
        self.model = model
        self.xcols = xcols
        self.ycols = ycols
        self.score_metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.X = dataframe.values[:, self.xcols].astype(int)
        self.y = dataframe.values[:, self.ycols].astype(int)
        self.train, self.test = train_test_split(dataframe)

        
    def get_X(self):
        """
        returns X feature array
        """
        return self.X
        
    
    def get_y(self):
        """
        returns y response array
        """
        return self.y
        
    
    def get_train(self):
        """
        returns training set
        """
        return self.train[:, self.xcols].astype(int), self.train[:, self.ycols].astype(int)
        
         
    def get_test(self):
        """
        returns test set
        """
        return self.test[:, self.xcols].astype(int), self.test[:, self.ycols].astype(int)
        
    
    def get_model(self):
        """
        returns estimator
        """
        return self.model
        
    
    def build_full_data_model(self):
        """
        return a fit of the estimator on the full data set
        """
        return self.model.fit(self.get_X(), self.get_y())
        
    
    def build_tt_model(self):
        """
        returns a fit of the estimator on the training set
        """
        return self.model.fit(self.get_train()[0], self.get_train()[1])
       
       
    def get_predictions(self):
        """
        return predictions and prediction probabilities based off of the test set
        """
        clf_model = self.build_tt_model()
        
        return clf_model.predict(self.get_test()[0]), \
                clf_model.predict_proba(self.get_test()[0])        
       
       
    def calculate_metrics(self, cv=False, tt=False):
        """
        return metrics either on cross validation or train/test set
        """
        if tt:        
            metric_scores = []
            
            metric_scores.append(accuracy_score(self.get_test()[1], self.get_predictions()[0]))
            metric_scores.append(precision_score(self.get_test()[1], self.get_predictions()[0]))
            metric_scores.append(recall_score(self.get_test()[1], self.get_predictions()[0]))
            metric_scores.append(f1_score(self.get_test()[1], self.get_predictions()[0]))
            
            return metric_scores
        
        if cv:
            return [np.mean(cross_val_score(self.get_model(), self.get_X(), self.get_y(), scoring=score)) \
                for score in self.score_metrics]
                    
    
    def print_metrics(self, cv=False, tt=False):
        """
        prints various metric scores
        """
        if cv:        
            metric_scores = self.calculate_metrics(cv=True)
            
            for i, metric in enumerate(self.score_metrics):
                print '{0} Score: {1:.2f}'.format(metric.capitalize(), metric_scores[i])
        if tt:
            metrics = self.calculate_metrics(tt=True)
        
            for i, metric in enumerate(metrics):
                print '{0} Score: {1:.2f}'.format(self.score_metrics[i].capitalize(), metric)
            
 
    def plot_learning_curve(self, model_name):
        """
        plots the learning curve based off the full data set
        """
        m, train_error, test_error = learning_curve(self.model, self.get_X(), self.get_y())
        plt.figure()
        plt.plot(m, np.mean(train_error, axis=1), label='Training')
        plt.plot(m, np.mean(test_error, axis=1), label='Test')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title(model_name + ' Learning Curve.')
        plt.legend(loc='best')
        
        
    def plot_ROC_curve(self, model_name, color):
        """
        plots the ROC curve based off of the test data
        """
        fpr, tpr, thresholds = roc_curve(self.get_test()[1], self.get_predictions()[1][:,1])
        AUC = auc(fpr,tpr)
        
        plt.plot(fpr, tpr, color, label=model_name + ': ' + str(AUC))
        plt.legend(loc='lower right') 
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')

