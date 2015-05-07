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
from sklearn import tree #import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
from sklearn.externals.six import StringIO  
import pydot

### Look into gradient boosting methods

## Setup dataframe 
bank_add = pd.read_csv('bank-additional.csv', sep=';') 
bank_add = bank_add.replace(to_replace=['yes','no'], value=[1,0])
#bank_add['cons.price.idx'] = preprocessing.scale(bank_add['cons.price.idx'])
bank_add_dum = pd.get_dummies(bank_add, columns=['loan', 'job', 'marital', 'education','default', 'housing', 'contact', 'poutcome'])

# Features = All columns
xcols = [0,4,5,6,7,8,9,10,11]
xcols.extend(range(13, bank_add_dum.shape[1]))
ycols = bank_add_dum.columns.get_loc('y')

## Data Exploration
#bank_dum.groupby('y').mean()
#plt.scatter(bank['scaled_balance'][bank['y'] == 1], bank['age'][bank['y'] == 1], c='r')
#plt.scatter(bank['scaled_balance'][bank['y'] == 0], bank['age'][bank['y'] == 0], c='y', alpha=0.1)

class Classifier(object):
    """
    builds a classifier object using an estimator model, a pandas dataframe, xcols 
    of features, and ycols for response label
    model = estimator
    dataframe = pandas dataframe
    xcols = tuple of integers
    ycols = integer
    """
    
    def __init__(self, model, dataframe, xcols, ycols):
        self.model = model
        self.xcols = xcols
        self.ycols = ycols
        self.score_metrics = ['accuracy', 'precision', 'recall', 'f1']
        self.X = dataframe.values[:, self.xcols].astype(float)
        self.y = dataframe.values[:, self.ycols].astype(float)
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
        return self.train.values[:, self.xcols].astype(float), \
                self.train.values[:, self.ycols].astype(float)
        
         
    def get_test(self):
        """
        returns test set
        """
        return self.test.values[:, self.xcols].astype(float), \
                self.test.values[:, self.ycols].astype(float)
        
    
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
        
        
def find_best_k(df, xcols, ycols):
    
    scores = []
    
    train, test = train_test_split(df)
    
    for i in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn = knn.fit(train.values[:, xcols].astype(int), train.values[:, ycols].astype(int))
        scores.append(knn.score(test.values[:, xcols].astype(int), test.values[:, ycols].astype(int))) 
    
    return scores.index(max(scores)) + 1
    
logistic = Classifier(LogisticRegression(), bank_add_dum, xcols, ycols)
svm = Classifier(SVC(probability=True), bank_add_dum, xcols, ycols)
knn = Classifier(KNeighborsClassifier(n_neighbors=find_best_k(bank_add_dum, xcols, ycols)), bank_add_dum, xcols, ycols)
gnb = Classifier(GaussianNB(), bank_add_dum, xcols, ycols)
dt = Classifier(tree.DecisionTreeClassifier(), bank_add_dum, xcols, ycols)
forest = Classifier(RandomForestClassifier(), bank_add_dum, xcols, ycols)
gradiant = Classifier(GradientBoostingClassifier(), bank_add_dum, xcols, ycols)
     
#dot_data = StringIO() 
#tree.export_graphviz(dt.build_full_data_model(), out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("bank.pdf")
