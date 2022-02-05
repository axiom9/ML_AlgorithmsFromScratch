# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 22:45:14 2022

@author: Anas
"""

#%% Imports

import numpy as np
from sklearn import datasets # For testing our model on regression data
from sklearn.model_selection import train_test_split # Testing our model
import matplotlib.pyplot as plt


#%% Linear Regression from scratch

class LinearRegression:
    
    def __init__(self, lr=.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        '''This method fits the model to the data via algorithims such as gradient descent
        "X" are the training samples, MxN ndarray (M = number of samples, N = number of features)'''
        #Initialize gradient descent parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) #Can also try random initialization between 0 - 1
        self.bias = 0
        
        #Gradient descent
        for _ in range(self.n_iters):
            
            y_pred = np.dot(X, self.weights) + self.bias
            
            # dw = (1/n_samples)*np.sum(np.multiply(X, (y_pred - y)))
            dw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    
    def predict(self, X):
        '''This method makes predictions using the updated parameters'''
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def metrics(self, y_true, y_pred):
        '''This method returns the metrics for this regression problem (MSE, MAE, RMSE)'''
        #RMSE
        return np.sqrt(np.mean((y_true-y_pred)**2))
    
    
    
#%% Testing

if __name__ == '__main__':
    print('Testing model...')
    
    def gen_plot(preds):
        '''This function is used to generate a plot with our model's best fit line'''
        # fig = plt.figure(figsize=(8,6))
        cmap = plt.get_cmap('viridis')
        m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
        m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
        plt.plot(X, y_preds, color='black', linewidth=2, label = 'Predictions')
        plt.show()
    
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=50, random_state=1)
    
    # X, y = datasets.load_diabetes(return_X_y = True)
    print(f'(Shape of X: {X.shape}\nShape of y: {y.shape})')

    #Visualize the data
    plt.scatter(X, y)
    plt.show()

    #Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=1)
    
    #Hyperparameters
    lr = .01
    n_iters = 1000
    regression = LinearRegression(lr=lr, n_iters=n_iters)
    regression.fit(X_train, y_train)
    
    # Make predictions
    preds = regression.predict(X_test)
    
    # Evaluate model performance
    print(f'RMSE: {regression.metrics(y_test, preds)}')
    print(f'MSE: {regression.metrics(y_test,preds)**2}')
    
    # Plotting
    y_preds = regression.predict(X)
    gen_plot(preds=y_preds)
    