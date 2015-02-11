# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:18:39 2015

@author: dsing001
"""
from sklearn.metrics import confusion_matrix, classification_report 
from NN4 import MyFirstNN
from sklearn import datasets 
import math
import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 100,fnc='sigmoid'): 
    
    lbls = set(np.hstack((test_class,train_class)))
    n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,Normalize=True,batch_size = 2,labels = lbls) 

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
    
dgts = datasets.load_digits()

dgts_data = dgts.images
dgts_labels = dgts.target


print (len(dgts_labels))
numofrows = dgts_data.shape[0]
dgts_data = dgts_data.reshape(numofrows,64)
#dgts_data = dgts_data[:300,:]
#dgts_labels = dgts_labels[:300]

num_fold = 2
gen_k_sets = StratifiedKFold(dgts_labels,num_fold,True)
gen_k_sets = StratifiedShuffleSplit(dgts_labels, n_iter=1, test_size=0.2)

for train_index, test_index in gen_k_sets:   
    train_data, test_data = dgts_data[train_index], dgts_data[test_index]
    train_class, test_class = dgts_labels[train_index], dgts_labels[test_index]

print (datetime.datetime.now())
train_neural_net(train_data,train_class,test_data,test_class,n_hidden_layer=20,learning_rate=1,epochs =10)
print (datetime.datetime.now())