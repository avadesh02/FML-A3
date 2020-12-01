## This is an implementation of the AdaBoost algorithm as described in 
## The text book Foundation of Machine Learning pg - 146
## The base classifier is assumed to be stumps
## Author : Avadesh Meduri
## Date : 29/11/2020

import numpy as np
from sklearn import tree
import torch

class AdaBoost:

    def __init__(self):
        pass

    def init(self, x_train, y_train):
        '''
        Input:
            x_train : train data
            y_train : labels of train data
        '''
        self.m = len(y_train) 
        self.D = np.ones(self.m)/self.m # probability distribution
        self.x_train = x_train
        self.y_train = y_train
        self.y_pred_arr = []
        self.h_arr = []
        self.alpha_arr = []

    def train(self, T):
        '''
        Input:
            T : number of training interations
        '''
        for t in range(T):
            # Defining the hypothesis set (stumps)
            h = tree.DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            h.fit(self.x_train, self.y_train, sample_weight = self.D)
            self.h_arr.append(h)
            
            y_predict = h.predict(self.x_train)
            self.y_pred_arr.append(y_predict)
            mistakes = (y_predict != self.y_train)
            eps_t = np.mean( np.average(mistakes, weights=self.D, axis=0))
            
            alpha_t = 0.5*np.log((1.0 - eps_t)/eps_t)
            self.alpha_arr.append(alpha_t)
            
            Z_t = 2*np.power(eps_t*(1 - eps_t),0.5)

            for i in range(self.m):
                self.D[i] = self.D[i]*np.exp(-alpha_t*self.y_train[i]*y_predict[i])/Z_t
            # for i in range(self.m):
            #     u = -self.y_train[i]*np.dot(self.alpha_arr,(np.array(self.y_pred_arr)[:,i]))
            #     self.D[i] = (1/self.m)*(np.exp(u))
            # print("done", t)
            self.D = self.D/np.sum(self.D) 

    def predict(self, X_test):
        y = np.zeros(len(X_test))
        for i in range(len(self.h_arr)):
            y += self.alpha_arr[i]*self.h_arr[i].predict(X_test)
        
        return np.sign(y)

    
    