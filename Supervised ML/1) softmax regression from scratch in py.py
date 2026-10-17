# import numpy as np

# y = np.array([0,1,2,2])

# def one_hot_encode(y):
#     n_col = len(np.unique(y))
#     n_row = y.shape[0]
#     y_encode = np.zeros((n_row, n_col))

#     for index, val in enumerate(y):
#         y_encode[index,val] = 1.0
#     return y_encode

# X = np.array([[0.1, 0.5],
#               [1.1, 2.3],
#               [-1.1, -2.3],
#               [-1.5, -2.5]])

# W = np.array([[0.1, 0.2, 0.3],
#               [0.1, 0.2, 0.3]])

# bias = np.array([0.01, 0.1, 0.1])

# print('Inputs X:\n', X)
# print('\nWeights W:\n', W)
# print('\nbias:\n', bias)

####################

# import some data to play with
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
X = iris.data
y = iris.target

# standardize the input features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#
from softmax import SoftmaxRegression