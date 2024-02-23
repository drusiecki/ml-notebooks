"""Some activation and loss functions along with their derivatives"""
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def simoid_dx(x):
    return np.exp(-x)/(1+np.exp(-x))**2

def relu(x):
    return (x > 0) * x

def relu_dx(x):
    return (x > 0)

def neg_log_loss(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    nll = - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    return np.sum(nll)

def neg_log_loss_dx(y_pred, y_true):
    return np.sum(-y_true/y_pred + (1-y_true)/(1-y_pred))

def mse(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    n_cases = y_pred.shape[0]
    return (y_pred - y_true) ** 2 / n_cases

def mse_dx(y_pred, y_true):
    n_cases = y_pred.shape[0]
    return 2 * (y_true - y_pred) / n_cases