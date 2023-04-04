# %%
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
# %%
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = np.trace(cm)/sum(cm.flatten())

    classwiseAccuracy = np.zeros(len(cm))
    for n in range(len(cm)):
        for i in range(len(cm)):
            for j in range(len(cm)):
                if (i != n and j != n) or (i == n and j == n):
                    classwiseAccuracy[n] += cm[i][j]

    classwiseAccuracy /= sum(cm.flatten())

    plt.figure(figsize=(6, 6))
    plt.title('Accuracy Score: {0:3.3f}'.format(overallAccuracy), size=12)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues')

    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))


# %%
# import dataset
iris = datasets.load_iris()

print(iris.data.shape)
print(iris.target.shape)

# %%
# train test split dataset
x_train, x_test, y_train, y_test = train_test_split(
    iris.data[0:100], iris.target[0:100], test_size=0.2, shuffle=True, random_state=42)


# %%
# data visualisation
seaborn.set(style='whitegrid')
seaborn.set_context('talk')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
plt.subplot(1, 2, 1)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm.viridis)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.subplot(1, 2, 2)
plt.scatter(x_test[:, 2], x_test[:, 3], c=y_test, cmap=cm.viridis)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

# %%


class Perceptron:
    def __init__(self, x_train, y_train):
        self.data = x_train
        self.y = y_train
        self.w = np.ones(4)

    def train(self, epoch):
        count = 1
        correctClassified = 0
        print(self.w)
        while (correctClassified != len(self.data) and count < epoch):  # Until everything is classified
            print(f'Iteration is : {count}')
            for sample in range(len(self.data)):
                x = np.append(self.data[sample, 0:3], 1)
                print(f'Sample is : {x}')
                print(f'Class is : {self.y[sample]}')
                wx = np.dot(np.transpose(self.w), x)
                print(f'Dot Product of W & X is : {wx}')
                if self.y[sample] == 1:  # Sample is positive
                    if wx >= 0:  # WX >= 0
                        correctClassified = correctClassified+1
                        print("Positive Sample is correctly clasified")
                    else:
                        print("Positive Sample is classified negative")
                        self.w = self.w+x
                        print(f'Updated W is : {self.w}')
                else:  # Sample is Negative
                    if wx < 0:  # WX < 0
                        correctClassified = correctClassified+1
                        print("Negative Sample is Correctly classified")
                    else:
                        print("Negative Sample is classified positive")
                        self.w = self.w-x
                        print(f'Updated W is : {self.w}')
            count = count+1
            if (correctClassified != len(self.data)):
                correctClassified = 0

            print(self.w)

    def predict(self, x_data):
        return np.where(np.dot(x_data, self.w.T) > 0, 1, 0)


# %%
iris_perceptron = Perceptron(x_train=x_train, y_train=y_train)
iris_perceptron.train(2)
y_pred = iris_perceptron.predict(x_test)


# %%
confusionMatrixAndAccuracyReport(y_test, y_pred)
