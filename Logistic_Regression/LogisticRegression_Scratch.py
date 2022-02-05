# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 21:35:01 2022

@author: Anas
"""
#%% Imports

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

# import pandas as pd
#%% Implementation

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        ''' X is MxN ndarray where M is the number of samples
        and N is the number of features. 'y' are the labels'''
        n_samples, n_features = X.shape
        self.weights=np.zeros(n_features) #Let's try random as well
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias #WX+b
            y_preds = self._sigmoid(linear_model)
            
            #Calculate derivatives
            dw = (1/n_samples) * np.dot(X.T, (y_preds - y))
            db = (1/n_samples) * np.sum(y_preds - y)
        
            # Update parameters
            self.weights -= self.lr* dw
            self.bias -= self.lr*db
            
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_out = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_out
    
    def _sigmoid(self, z):
        return (1 / (1 + np.exp(-z))).astype('float64')
    
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
    
#%% Testing our implementation
if __name__ == "__main__":
    
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1)
    
    
    #Hyperparameters
    lr = .001
    n_iters = 1000

    classifier = LogisticRegression(lr=lr, n_iters=n_iters)

    classifier.fit(X_train, y_train)
    
    preds = classifier.predict(X_test)
    
    print('Classification Accuracy: {}'.format(classifier.accuracy(y_test, preds)))
        