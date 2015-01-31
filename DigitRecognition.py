# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:18:39 2015

@author: dsing001
"""
from sklearn.metrics import confusion_matrix, classification_report 
from NeuralNet2 import MyFirstNN
from sklearn import datasets 
import math
import datetime
    
dgts = datasets.load_digits()

dgts_data = dgts.images
dgts_labels = dgts.target

print (len(dgts_labels))

train_data = dgts_data[:1000,:,:]
test_data = dgts_data[1000:,:,:]
train_labels = dgts_labels[:1000]
test_labels = dgts_labels[1000:]


n_in_layer = 64
n_hidden_layer = 100
n_outer_layer = 10

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 10,learning_rate=1,epochs = 100,fnc='sigmoid'):
 
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
train_neural_net(train_data,train_labels,test_data,test_labels,n_hidden_layer=50,learning_rate=1.0,epochs =200)
print (datetime.datetime.now())