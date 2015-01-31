# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 19:04:21 2014

@author: Darshan Singh

This is single hidden layer model only. It can be made generic to have as many as hidden layer and finally sum of signals at each node can be calculated 
using numpy matrix multiplication to make things bit easy.
"""
import numpy
import math


def tanh(x):
    return ((1 - math.exp(-2*x))/(1 + math.exp(-2*x)))
def tanh_diff(x):
    return (1 - tanh(x)**2)
def sigmoid(x):
    return (1/(1+math.exp(-x)))

def sigmoid_diff(x):
    return sigmoid(x)* (1-sigmoid(x))

def linear(x):
    return x
    
def linear_diff(x):
    return 1
    
class MyFirstNN:
    def __init__(self,n_in_layer,n_hidden_layer,n_outer_layer,fnc,fnc_diff,learning_eta= 1.0,epochs= 1000):
        self.n_in_layer = n_in_layer
        self.n_hidden_layer = n_hidden_layer
        self.n_outer_layer = n_outer_layer
        
        ###create separate weight matrix for input and hidden
        self.w_in_to_hidden = numpy.random.random((self.n_hidden_layer,self.n_in_layer))
        self.delta_w_in_to_hidden = numpy.zeros((self.n_hidden_layer,self.n_in_layer))
        self.w_hidden_to_out = numpy.random.random((self.n_outer_layer,self.n_hidden_layer))
        self.delta_w_hidden_to_out = numpy.zeros((self.n_outer_layer,self.n_hidden_layer))
        
     
        ''' You might want to use a different function at different hidden or output node '''
        self.fnc_hidden_layers = fnc[0]
        self.fnc_diff_hidden_layers = fnc_diff[0]
        self.fnc_output_layers = fnc[1]
        self.fnc_diff_output_layers = fnc_diff[1]
        self.learning_eta = learning_eta
            
            
        ''' store output of each hidden and output neuron '''
        self.out_layer_output= numpy.zeros(self.n_outer_layer)
        self.hidden_layer_out= numpy.zeros(self.n_hidden_layer)
        
        ''' store hidden layer and output layer differentiation values '''
        
        self.out_layer_diff= numpy.zeros(self.n_outer_layer)
        self.hidden_layer_diff= numpy.zeros(self.n_hidden_layer)
        self.out = numpy.zeros(self.n_outer_layer)
        
        self.epochs = epochs
        self.training_error= []

    def forward_pass(self):
        
        ''' visit all hidden nodes and calculate the outputs which will serve as input to output neurons '''
        for i in range(self.n_hidden_layer):            
            fnc_input = 0
            ''' all weights and inputs that is coming to this hidden layer neuron '''
            wghts = self.w_in_to_hidden[i]
            inpts = self.input_from_in_to_hidden

            for j in range(len(inpts)):   
                fnc_input = fnc_input + wghts[j]*inpts[j]
            '''store the hidden layer neuron output as well as hidden layer diff for that output '''
            
            self.hidden_layer_out[i] = self.fnc_hidden_layers[i](fnc_input)
            self.hidden_layer_diff[i] = self.fnc_diff_hidden_layers[i](fnc_input)
        ''' visit all output nodes and calculate the outputs which will then be compared with target to get error at that output node '''
        for i in range(self.n_outer_layer):    
            fnc_input = 0    
            ''' all weights and inputs that is coming to this output layer neuron from hidden layer'''
            wghts = self.w_hidden_to_out[i]   
            for j in range(self.n_hidden_layer):
                fnc_input = fnc_input + wghts[j]*self.hidden_layer_out[j]

             
            '''store the output layer neuron output as well as output layer diff for that output '''
            self.out_layer_output[i] = self.fnc_output_layers[i](fnc_input)
            self.out_layer_diff[i] = self.fnc_diff_output_layers[i](fnc_input)
  
    def backpropagate(self):
        ''' betas for output layer neuron to calculate the weights '''
        
        out_layer_betas = numpy.zeros(self.n_outer_layer)
        for i in range(self.n_outer_layer):
            ''' it is diff in target and calculated out multiply by diff of the function with output value as x'''
            out_layer_betas[i]= (self.out[i] - self.out_layer_output[i])*self.out_layer_diff[i] 
            ''' calculate all deltas for all weights coming from hidden layers '''
            for j in range(self.n_hidden_layer):
                self.delta_w_hidden_to_out[i][j] = self.learning_eta * out_layer_betas[i] * self.hidden_layer_out[j]
                        
                        
        ''' calculate hidden layer betas and  update input to hidden layer weights based on these betas '''
        hidden_layer_betas = numpy.zeros(self.n_hidden_layer)
       
        for i in range(self.n_hidden_layer):
            c= 0
            ''' one hidden layer input can go to various output and thus hiddn layer betas will be calculated base don these all outputs '''
            for j in range(self.n_outer_layer):
                c = c+ out_layer_betas[j] *self.w_hidden_to_out[j][i]
            hidden_layer_betas[i]= self.hidden_layer_diff[i] *c
            ''' calculate  the input to hidden weights delta '''
            for j in range(self.n_in_layer):
                self.delta_w_in_to_hidden[i][j] = self.learning_eta * hidden_layer_betas[i] * self.input_from_in_to_hidden[j]

       
    def update_weights(self):
        ''' update in to hidden weights '''
   
        for i in range(self.n_hidden_layer):
            for j in range(len(self.w_in_to_hidden[i])):
               self.w_in_to_hidden[i][j] = self.w_in_to_hidden[i][j] + self.delta_w_in_to_hidden[i][j]
        ''' update hidden to out weights '''
        for i in range(self.n_outer_layer):
            for j in range(len(self.w_hidden_to_out[i])):
                self.w_hidden_to_out[i][j] = self.w_hidden_to_out[i][j] + self.delta_w_hidden_to_out[i][j]
   
    def fit(self,X,Y):       
        for epoch in range(self.epochs):
            error= numpy.zeros(len(self.out))
            for i,inputs in enumerate(X):       
                   targets = Y[i]
                   self.input_from_in_to_hidden = inputs            
                   self.out = targets     
                   self.forward_pass()  
                   self.backpropagate()      
                   self.update_weights()
                   for i in range(len(targets)):
                       error[i] = error[i] + math.pow((targets[i]- self.out_layer_output[i]),2)
            for i in range(len(error)):
                error[i] = error[i]/len(X)
                self.training_error.append(error)
        train_err = self.training_error
        return(train_err)

       
    def predict(self,X):
       tst_cls_pred = []
       for inputs in X:  

           self.input_from_in_to_hidden = inputs 
           self.forward_pass() 
           tst_cls_out = self.out_layer_output.copy()

           tst_cls_pred.append(tst_cls_out)


       return tst_cls_pred

        
                
if __name__ == "__main__":
    def train_neural_net(train_data,train_cls,test_data,test_cls,learning_rate=1,epochs = 100,afnc=sigmoid,afnc_diff=sigmoid_diff):
        '''in below you might need to change the inner layer and hidden layer and outer layer values and weights and functions 
        to train neural netwrok in your own style 
        '''
        n_in_layer = 2
        n_hidden_layer = 3
        n_outer_layer = 1
     
        
        '''' you can assign differnt functions to different nodes by putting them in array based on neuron position '''
        fnc= []
        ''' these are corresponding neuron diff's '''
        fnc_diff = []
        fnc_list = []
        fnc_diff_list = []
        for i in range(n_hidden_layer):
            fnc_list.append(afnc)
            fnc_diff_list.append(afnc_diff)
        fnc.append(fnc_list)
        fnc_diff.append(fnc_diff_list)
        fnc.append([linear])
        fnc_diff.append([linear_diff])
     
        n = MyFirstNN(n_in_layer,n_hidden_layer,n_outer_layer,fnc,fnc_diff,learning_rate,epochs=epochs)
     
        
        train_err = n.fit(train_dt,train_cls)
        
        print (train_err[:30])
        predicted = n.predict(test_dt)
        print (predicted)
        
    
    train_dt = [[1,0],[1,1]]
    train_cls = [[0],[1]]
    test_dt = [[1,1],[1,0]]
    test_cls = [[1],[0]]
    train_neural_net(train_dt,train_cls,test_dt,test_cls,epochs =1000)
        
    
        
        
            
        
            