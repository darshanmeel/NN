# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:50:31 2015

@author: dsing001
"""

from sklearn.datasets import fetch_mldata


from sklearn.metrics import confusion_matrix, classification_report 
from NeuralNet2 import MyFirstNN
from sklearn import datasets 
import math
import datetime
    
mnist = fetch_mldata('MNIST original')

dgts_data = mnist.data
dgts_labels = mnist.target

print (len(dgts_labels))

train_data = dgts_data[:2000,:]
test_data = dgts_data[2000:,:]
train_labels = dgts_labels[:2000]
test_labels = dgts_labels[2000:]

for i in train_data[1,]:
    print (i)
print (g)
n_in_layer = 784
n_hidden_layer = 20
n_outer_layer = 10

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 10,learning_rate=1,epochs = 100,fnc='tanh'):
 
    n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,Normalize=True)
 
    
    train_err = n.fit(train_data,train_cls)

    print ('predict')
    #predicted = n.predict_proba(test_data)
    
    predicted = n.predict(test_data)
    correct = 0    
    for i,val in enumerate(test_cls):
        if predicted[i]==val:
            correct= correct + 1
    
    print (correct,correct*1.0/(len(test_cls)))
    print confusion_matrix(test_cls,predicted)  
    print classification_report(test_cls,predicted)
    

print (datetime.datetime.now())
train_neural_net(train_data,train_labels,test_data,test_labels,n_hidden_layer=30,learning_rate=1.0,epochs =1)
print (datetime.datetime.now())