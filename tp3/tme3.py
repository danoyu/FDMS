# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:03:28 2017

@author: 3202199
"""

import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

data = datasets.load_iris()
X = data.data
Y = data.target



class LinearModule():
    def __init__(self, dimension):
        self.dimension = dimension
        self.theta = np.random.rand(dimension)
        self.cost=[]
        self.accuracy=[]
       
    def randomize(self,variance):
        self.theta=(self.theta*2*variance)-variance
        
    def init_gradient(self):
        self.gradient=np.zeros((1,self.X))
        
    def update_theta(self,X,Y,idx,step):
        tmp_theta = np.zeros(self.dimension)
        for k in range(self.dimension):
            tmp_theta[k] = self.theta[k] - step*2*X[idx][k]*(X[idx][k]*self.theta[k] - Y[idx]) 
            if tmp_theta[k] < 0:
                tmp_theta[k] -= coeff
            else :
                tmp_theta[k] += coeff
        return tmp_theta
    
    def fit(self,X,Y,max_iter,step):
        for it in range(max_iter):
            tmp_cost=0
            for i in range(len(X)):
                idx = np.random.randint(0,len(X)-1)
                tmp_theta = self.update_theta(X,Y,idx,step)
                for j in range(self.dimension):
                    if (tmp_theta[j]*self.theta[j])<0:
                        self.theta[j]=0
                    else:   
                        self.theta[j]=tmp_theta[j]
                tmp_cost+=(Y[i]-np.dot(X[i],self.theta))*(Y[i]-np.dot(X[i],self.theta))
            
            self.cost.append(tmp_cost/len(X) + coeff*np.sum(np.absolute(self.theta)))
            
    def predict(self, X):
        return np.dot(X,self.theta)
        
        
        

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
max_iter = 1000
coeff = 0
step = 0.00001

model = LinearModule(4)
model.randomize(0.1)

model.fit(X_train,Y_train,max_iter,step)

plt.plot(range(max_iter),model.cost)






