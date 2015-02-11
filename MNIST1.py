# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:50:31 2015

@author: dsing001
"""

from sklearn.datasets import fetch_mldata
from sklearn.metrics import confusion_matrix, classification_report 
from NN4 import MyFirstNN
from sklearn import datasets 
import math
import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np

import random 

import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


'''
mnist = fetch_mldata('MNIST original')

dgts_data = mnist.data
dgts_labels = mnist.target
'''

dgts_data = np.array(train_set[0])
dgts_labels = np.array(train_set[1])

test_data = np.array(valid_set[0])
test_class = np.array(valid_set[1])

print (len(dgts_labels))

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 100,fnc='sigmoid'): 
    
    lbls = set(np.hstack((test_cls,train_cls)))
    n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,Normalize=True,batch_size = 10,labels = lbls) 

    train_err,wghts_after_each_epoch = n.fit(train_data,train_cls)
    print train_err
    #print wghts_after_each_epoch
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
    

numofrows = dgts_data.shape[0]
dgts_data = dgts_data.reshape(numofrows,784)
#dgts_data = dgts_data[:300,:]
#dgts_labels = dgts_labels[:300]

print (datetime.datetime.now())
train_neural_net(dgts_data,dgts_labels,test_data,test_class,n_hidden_layer=100,learning_rate=0.99,epochs =100)
print (datetime.datetime.now())