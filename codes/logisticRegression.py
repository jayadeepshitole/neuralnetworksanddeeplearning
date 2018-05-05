# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:48:56 2018

@author: jayadeep
"""

import numpy as np
from generalFunctions import sigmoid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

class LogisticRegression(object):

    ## Initialize the parameters    
    def __init__(self, dim):
        """
        This function creates vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        """
        self.dim = dim
        self.weight = np.zeros((dim, 1))
        self.bias = 0
    
    ## Forward and Backward Propagation
    def propagate(self, X, Y):
        """
        Imlement the cost function and its gradient for the propagation explained.
        
        Arguments:
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, and 1 if cat) of shape (1, number of examples)
        
        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b        
        """
    
        m = X.shape[1]
        
        # FORWARD PROPAGATION
        Z = np.dot(self.weight.T, X) + self.bias
        A = sigmoid(Z)
        cost = (-1 / m) * (np.dot(Y,( np.log(A).T)) + np.dot(1 - Y, (np.log(1 - A)).T)) 
        
        # BACKWARD PROPAGATION
        dZ = A - Y
        dw = (1 / m) * np.dot(X, dZ.T)
        db = (1 / m) * np.sum(dZ)
        
        assert (dw.shape == self.weight.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        
        grads = {"dw": dw, 
                 "db": db}
        
        return grads, cost
    
    def train(self, X, Y, num_iterations = 1000, learning_rate = 0.01, print_cost = False, plot_cost = False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, and 1 if cat) of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop (default = 1000)
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print cost every 100 steps
        
        Returns:
        grads -- dictionary containing the gradients of the weights and biases with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve
        """
        costs = []
        
        for i in range(num_iterations + 1):
            grads, cost = self.propagate(X,Y)
            
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            # Update rule
            self.weight = self.weight - learning_rate * dw
            self.bias = self.bias - learning_rate * db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training examples
            if print_cost and i % 100 == 0:
                print(f"cost after iteration: {i} is {np.round(cost, 4)}")
        
        grads = {"dw": dw,
                 "db": db}

        if plot_cost:
            plt.plot(np.arange(len(costs)) * 100, costs)
            plt.xlabel("epoch")
            plt.ylabel("cost")
            plt.title(f"Logistic curve fitting loss function (learning rate={learning_rate})")
            plt.show()
            
        return grads, costs


    
    def predict(self, X):
        """
        compute predicted value of class probability using fitted looistic regression model
        
        Argument:
        X -- data a numpy ndarray of shape (num_features, number of examples)
        
        Return:
        p -- a numpy ndarray of shape (1, number of examples), prediction of prob(class = 1)
        """
        
        p = sigmoid(np.matmul(self.weight.T , X)  + self.bias)
        return p
    
    def accuracy_stats(self, y_true_label, y_pred_prob, threshold = 0.50, accuracy = True, confusion_mat = True, ROC = True):
        """
        Compute accuracy of predictions - 1. accuracy 2. confusion matrix  
        
        Arguments:
        y_true_label -- true "labels" y, a numpy array of shape (1, number of examples) or (number of examples, 1) or (number of examples, )
        y_pred_prob -- prediction of probability y = 1, a numpy array of same shape as y_true_label
        threshold -- y = 1 if prob(y = 1) >= threshold (defaulf = 0.50)
        accuracy -- True prints accuracy of the prediction
        confusion_mat -- True prints the confusion matrix    
        """
        # Reshape 
        y_true_label = np.squeeze(y_true_label)
        y_pred_prob = np.squeeze(y_pred_prob)
        
        y_pred_label = (y_pred_prob >= threshold) + 0
        acc = np.mean(1 - abs(y_true_label - y_pred_label))
        
        #print("\n --------PRINTING ACCURACY STATISTICS-------- \n")
        
        if accuracy:
            print(f"accuracy: {acc * 100}%\n")
        
        if confusion_mat:
            print("confusion matrix:\n" + str(confusion_matrix(y_true_label, y_pred_label)) + "\n")
        
        if ROC:
            fpr, tpr, thresholds = roc_curve(y_true_label, y_pred_label)
            roc_auc = roc_auc_score(y_true_label, y_pred_label)
            
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.show()
            print("ROC AUC:\n" + str(roc_auc) + "\n")
        
        
        
        
        
        
        