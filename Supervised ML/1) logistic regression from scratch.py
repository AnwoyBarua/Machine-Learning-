# import necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  

X,y = make_classification(n_features = 4,n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

# create a function that fits the training dataset and finds the value of parameters using gradient ascent
# input: X_train, y_train, learning_rate, iteration.
# Y = (m,1) ; X = (m, n) ; theta = (n,1)
# output: vector of parameters

###### There is a major problem with the following code:

# the following function uses gradient ascent algorithm.

# def fit(X, Y, alpha = 0.001, iter = 400):
#     _, n = X.shape
#     theta = np.zeros((n, 1)) # (n, 1) vector
#     Y = np.reshape(Y, (len(Y), 1)) 

#     def sigmoid(X_matrix, theta):
#         m,_ = X_matrix.shape
#         out = []
#         for i in range(m):
#             multiply = np.dot( X_matrix[i, ] , theta) # gives a single output --> (1,1) = (1,n) *(n,1)
#             sigmoid_fn = 1/(1+ np.exp(-multiply))
#             out.append(sigmoid_fn)
#         out = np.reshape(out, (len(out), 1))
#         return out # returns an (m,1) vector.

#     for i in range(iter):

#         theta = theta + alpha * np.dot(X.T, Y - sigmoid(X, theta) )

#     return theta

# def predict(X, theta):
    
#     return X @ theta >= 0

# theta_new = fit(X_train, y_train)
# print(theta_new)
# # print(predict(X_train, theta_new))
# print(np.mean(predict(X_train, theta_new) == y_train))



############ logistic regression using newton's method.

class LogisticRegression():

    def __init__(self, theta = None, eps = 1e-6):
        self.theta = theta
        self.eps = eps
        
    def fit(self,x,y, alpha = 0.01):
        m,n = x.shape
        def sigmoid(z):
            return 1/(1+np.exp(-z))
        
        def gradient(x,y, theta):
            z = np.dot(x, theta)
            h_x = sigmoid(z) ## gives an (m,1) vector
            return (-1/m) * np.dot(x.T, (y - h_x)) ## (n,m) * (m,1) --> (n,1)

        def hessian(x, theta):
            z = np.dot(x, theta)
            h_x = sigmoid(z)
            h_x = np.reshape(h_x, (-1,1))
            return (1/m) * np.dot(x.T, h_x * (1-h_x) * x) ## should return a (n,n) vector.
        
        def next_theta(x,y, theta):
            H = hessian(x, theta)
            Grd = gradient(x,y,theta)
            return theta - np.dot(np.linalg.inv(H), Grd)
        
        if self.theta is None:
            self.theta = np.zeros(n)
        old_theta = self.theta
        new_theta = next_theta(x,y, old_theta)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(x, y, old_theta)
        self.theta = new_theta

    def predict(self, x):
        return x @ self.theta >= 0
        
    def accuracy(self):
        return np.mean(self.predict(self.x) == self.y)