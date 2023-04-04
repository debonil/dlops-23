# %%
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# # Question 2.
# ## MLP implementation

# %%


def pre_process_data(X_train_original, Y_train, X_test_original, Y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_original)

    mean = X_train_original.mean(axis=0)
    std = X_train_original.std(axis=0)

    mean = mean[:, None]
    std = std[:, None]

    X_train = X_train.transpose()

    X_test = X_test_original.transpose()

    Y_train = Y_train.transpose()
    Y_test = Y_test.transpose()

    print(X_train.shape)
    return X_train, Y_train, X_test, Y_test, mean, std


class MultiLayerPerceptron:

    def __init__(self, layer_nodes):
        self.L = len(layer_nodes)

        self.parameters = {}

        for l in range(1, self.L):
            self.parameters['W'+str(l)] = np.random.randn(layer_nodes[l],
                                                          layer_nodes[l-1])
            self.parameters['b'+str(l)] = np.zeros([layer_nodes[l], 1])

    def standardize(self, X, mean, std):
        X = (X-mean)/std
        return X

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def deriv_relu(self, x):
        x = (x >= 0)
        return x

    def deriv_sigmoid(self, x):
        m = self.sigmoid(x)
        return m*(1-m)

    def forward_propagation(self, X):
        cache = {"A0": X, }
        L = len(self.parameters)//2
        for l in range(1, L+1):
            cache['Z'+str(l)] = np.dot(self.parameters['W'+str(l)],
                                       cache['A'+str(l-1)])+self.parameters['b'+str(l)]
            cache['A'+str(l)] = self.relu(cache['Z'+str(l)])
        return cache

    def compute_cost(self, Yhat, Y):
        return np.mean(np.power((Yhat-Y), 2))

    def backward_propagation(self, parameters, cache, Y):
        m = Y.shape[1]
        L = len(parameters)//2
        gradients = {}
        dA = 2*(cache['A'+str(L)]-Y)
        for l in range(0, L):
            dZ = dA*self.deriv_relu(cache['Z'+str(L-l)])
            gradients['dW'+str(L-l)] = (1/m) * \
                np.dot(dZ, cache['A'+str(L-l-1)].transpose())
            gradients['db'+str(L-l)] = (1/m)*np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(parameters['W'+str(L-l)].transpose(), dZ)
        return gradients

    def updation(self, parameters, gradients, learning_rate=0.0025):
        L = len(parameters)//2
        for l in range(1, L+1):
            parameters['W'+str(l)] = parameters['W'+str(l)] - \
                learning_rate*gradients['dW'+str(l)]
            parameters['b'+str(l)] = parameters['b'+str(l)] - \
                learning_rate*gradients['db'+str(l)]
        return parameters

    def model_training(self, X_train, Y_train, mean, std):
        self.mean = mean
        self.std = std
        for epoch in range(10000):

            cache = self.forward_propagation(X_train)
            # print(f'epoch #{epoch}: cache => {cache.keys()}')
            gradients = self.backward_propagation(
                self.parameters, cache, Y_train)
            self.parameters = self.updation(self.parameters, gradients)

            if (epoch % 1000 == 0):
                cost = self.compute_cost(cache["A3"], Y_train)
                print(cost)

    def predict(self, X):
        X_test = self.standardize(X, self.mean, self.std)
        output = self.forward_propagation(X_test)
        return output["A"+str(self.L-1)]

# %% [markdown]
# ### MLP driver code


# %%

X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target
Y = Y[:, None]


X_train_original, X_test_original, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, random_state=True)
X_train, Y_train, X_test, Y_test, mean, std = pre_process_data(
    X_train_original, Y_train, X_test_original, Y_test)

# Defining Layers
mlp_classifier = MultiLayerPerceptron([X_train.shape[0], 4, 5, 3])


# %%
mlp_classifier.model_training(X_train, Y_train, mean, std)


final = mlp_classifier.forward_propagation(X_train)
ycap = final["A3"]
m = mlp_classifier.compute_cost(ycap, Y_train)
print("Train accuracy", 1-m)

Yhat = mlp_classifier.predict(X_test_original.T)
c = mlp_classifier.compute_cost(Yhat, Y_test)
print("Test Accuracy", 1-c)

# %% [markdown]
# #### Final Weightages and biases

# %%
mlp_classifier.parameters

# %%
confusionMatrixAndAccuracyReport(Y_test.T.squeeze(), np.argmax(Yhat.T, axis=1))
