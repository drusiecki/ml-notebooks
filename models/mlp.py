from models.func import mse, mse_dx, neg_log_loss, neg_log_loss_dx, relu, relu_dx, sigmoid, simoid_dx
import numpy as np
from models.func import *


class MLP:
    """A simple implementation of an MLP or a straight-forward, feed-forward neural network"""

    _act_f_dict = {
        "SIGMOID": (sigmoid, simoid_dx),
        "RELU" : (relu, relu_dx),
        "NONE": (lambda x: x, lambda x: 1)
    }
    
    _loss_dict = {
        "SIGMOID": (neg_log_loss, neg_log_loss_dx),
        "RELU": (mse, mse_dx),
        "NONE": (mse, mse_dx)
    }
    
    def __init__(self, layer_widths, act_f="SIGMOID"):
        self.actf, self.actfd = self._act_f_dict[act_f] # we assume the same act-f for every layer for now
        self.loss_function, self.loss_function_dx = self._loss_dict[act_f]
        n_layers = len(layer_widths) - 1
        self.W = [ np.random.randn(layer_widths[i], layer_widths[i+1]) for i in range(n_layers) ]
        
    # input should be of shape (n_cases, first_layer_width)
    # output is of shape (n_cases, last_layer_width)
    def _forward_pass(self, X, intermediate=False):
        layer_activations = [X]
        layer_outputs = []
        
        layer_input = X
        for w in self.W:
            layer_output = np.matmul(layer_input, w)
            layer_activation = self.actf(layer_output)
            if intermediate:
                layer_outputs.append(layer_output)
                layer_activations.append(layer_activation)
            layer_input = layer_activation
        if intermediate:
            return layer_input, layer_outputs, layer_activations[:-1]
        else:
            return layer_input
        
    def predict(self, X):
        return self._forward_pass(X)
    
    def loss(self, X, y_true):
        y_pred = self._forward_pass(X)
        return self.loss_function(y_pred, y_true)
    
    def loss_dx(self, y_pred, y_true):
        return self.loss_function_dx(y_pred, y_true)
    
    def _shuffle(self, X):
        np.random.shuffle
    
    def train(self, X, Y, iters=100, lrate=0.01):
        for i in range(iters):
            self._learn(X, Y, lrate)
#             print(f"Loss at {i}: {self.loss(X, Y)}")

    def train_batch(self, X, Y, iters=100, lrate=0.01, batch_size=100):
        x_index = 0
        y_index = 0
        Y = Y.reshape(-1, 1)
        for i in range(iters):
            X_batch, x_index = self.get_next_batch(X, x_index, batch_size)
            y_batch, y_index = self.get_next_batch(Y, y_index, batch_size)
            self._learn_batch(X_batch, y_batch, lrate)
            
    def train_stochastic_batch(self, X, Y, iters=100, lrate=0.01, batch_size=100):
        Y = Y.reshape(-1, 1)
        for i in range(iters):
            X_batch = self.get_stochastic_batch(X, batch_size)
            y_batch = self.get_stochastic_batch(Y, batch_size)
            self._learn_batch(X_batch, y_batch, lrate)
    
    def _learn(self, X, Y, lrate=0.01):
        for x, y in zip(X, Y):
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            # self._learn_batch(x, y, lrate)
            y_pred, layer_outputs, layer_activations = self._forward_pass(x, intermediate=True)
            
            nll_gradient = -y/y_pred + (1-y)/(1-y_pred)
            d = nll_gradient * self.actfd(layer_outputs[-1])
            
            W_grads = []
            for i in range(len(self.W)-1, -1, -1):
                W_grads.append(np.outer(layer_activations[i].T, d))
                d = np.matmul(d, self.W[i].T) * self.actfd(layer_outputs[i-1])
                
            W_grads = W_grads[::-1]
            self.W = [W - lrate*grad for W, grad in zip(self.W, W_grads)]
            
    def get_next_batch(self, X, index, batch_size):
        n_cases = X.shape[0]
        index_to = min(index + batch_size, n_cases) 
        batch = X[index:index_to,:]
        new_index = index_to % n_cases
        return batch, new_index
    
    def get_stochastic_batch(self, X, batch_size):
        indices = np.random.choice(X.shape[0], batch_size, replace=False)
        return X[indices,:]
    
    # X is (n_cases, n_features), Y is (n_cases, n_outputs)
    # whole dataset is computed at once
    # cost function is negative log-likelihood
    def _learn_batch(self, X, Y, lrate=0.01):
        n_cases = X.shape[0]
        Y = Y.reshape(-1, 1)
        y_pred, layer_outputs, layer_activations = self._forward_pass(X, intermediate=True)

        loss_function_derivative = self.loss_function_dx(y_pred, Y)
        d = loss_function_derivative * self.actfd(np.sum(layer_outputs[-1], axis=0).reshape(1, -1))

        W_grads = []
        for i in range(len(self.W)-1, -1, -1):
            W_grads.append(np.outer(np.sum(layer_activations[i], axis=0).reshape(1, -1).T, d))
            d = np.matmul(d, self.W[i].T) * self.actfd(np.sum(layer_outputs[i-1], axis=0).reshape(1, -1))
            
        W_grads = W_grads[::-1]
        self.W = [W - lrate*grad for W, grad in zip(self.W, W_grads)]