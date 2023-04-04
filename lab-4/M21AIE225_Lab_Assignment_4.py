# %% [markdown]
# # DL/DLOps (2023) - Lab Assignment 4: RNN
# ## Submitted by - Debonil Ghosh [M21AIE225]

# %%
import seaborn as sns
import math
import time
from sklearn.model_selection import train_test_split
import string
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import tqdm as notebook_tqdm
from sklearn import metrics

# %%
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

path = '../downloads/language_names'


def findFiles(path): return glob.glob(path)


print(findFiles(path+'/names/*.txt'))


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles(path+'/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print(f'Data loaded : Category size: {len(all_categories)}')


# %%

# Find letter index from all_letters, e.g. "a" = 0

def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


print(letterToTensor('J'))

print(lineToTensor('Jones'))
print(lineToTensor('Jones').size())

# %%

all_data = []
labels = []

for i, c in enumerate(all_categories):
    print(f'category: {c} | category_lines size: {len(category_lines[c])}')
    for n in category_lines[c]:
        all_data.append(lineToTensor(n))
        labels.append(torch.tensor([i], dtype=torch.long))


# %% [markdown]
# # 1. Split the data into train, val, and test (80:10:10).

# %%

#all_data = np.array(all_data)
#labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(
    all_data, labels, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, random_state=42)

print(f'training size: {len(x_train)}')
print(f'validation size: {len(x_val)}')
print(f'test size: {len(x_test)}')

# %% [markdown]
# # RNN Model Architechture

# %%


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, output_size, out_softmax=True):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.out_softmax = out_softmax

        self.i2lh = nn.Linear(input_size + hidden_size, linear_size)
        self.l2h = nn.Linear(linear_size, hidden_size)
        self.i2lo = nn.Linear(input_size + hidden_size, linear_size)
        self.l2o = nn.Linear(linear_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2lh(combined)
        hidden = self.tanh(hidden)
        hidden = self.l2h(hidden)
        hidden = self.tanh(hidden)
        output = self.i2lo(combined)
        output = self.tanh(output)
        output = self.l2o(output)
        output = self.softmax(
            output) if self.out_softmax else self.tanh(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def evaluate(self, line_tensor):
        hidden = self.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

        return output

    def predict(self, input_line, n_predictions=3):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = self.evaluate(lineToTensor(input_line))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])


# %%
def plot_training_graphs(loss_list):
    fig = plt.figure(figsize=(20, 7))
    plot = fig.add_subplot(1, 2, 1)
    plot.set_title("Training vs Validation loss")
    plot.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
    plot.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Loss")
    plot.legend()
    plot = fig.add_subplot(1, 2, 2)
    plot.set_title("Training vs Validation Accuracy")
    plot.plot(loss_list[:, 2], linestyle='--', label="Training Accuracy")
    plot.plot(loss_list[:, 3], linestyle='-', label="Validation Accuracy")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Accuracy")
    plot.legend()
    plt.show()


# %%


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# %%
# Train the model


def model_training(rnn, criterion, optimizer, x_train, y_train, x_val, y_val, num_epochs=10):
    start = time.time()
    loss_list = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        for i, line_tensor in enumerate(x_train):
            category_tensor = y_train[i]
            hidden = rnn.initHidden()
            rnn.zero_grad()
            # Forward pass
            for i in range(line_tensor.size()[0]):
                outputs, hidden = rnn(line_tensor[i], hidden)

            loss = criterion(outputs, category_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += metrics.accuracy_score(
                category_tensor, outputs.argmax(axis=1))
        # Evaluate the model on the validation set
        with torch.no_grad():
            #images, labels = next(iter(testloader))
            for i, line_tensor in enumerate(x_val):
                outputs = rnn.evaluate(line_tensor)
                loss = criterion(outputs, y_val[i])
                val_loss += loss.item()
                val_acc += metrics.accuracy_score(
                    y_val[i], outputs.argmax(axis=1))
        train_loss = train_loss/len(x_train)
        val_loss = val_loss/len(x_val)
        train_acc = train_acc/len(x_train)
        val_acc = val_acc/len(x_val)
        print(f'Epoch: {epoch+1} ({timeSince(start)}) \tTraining Loss: {train_loss:.3f}, \tTest Loss: {val_loss:.3f},  \tTraining acc: {train_acc:.2f}, \tTest acc: {val_acc:.2f}, ')
        loss_list.append([train_loss, val_loss, train_acc, val_acc])

    print(
        f'Training completed in {timeSince(start)} \tTraining Loss: {loss_list[-1][0]:.3f}, \tTest Loss: {loss_list[-1][1]:.3f},  \tTraining acc: {loss_list[-1][2]:.2f}, \tTest acc: {loss_list[-1][3]:.2f}, ')
    return np.array(loss_list)


# %%
#sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred_probs, label):
    Y_pred = Y_pred_probs.argmax(axis=1)
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = metrics.accuracy_score(Y_test, Y_pred)

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    top_5_accuracy = metrics.top_k_accuracy_score(
        Y_test, Y_pred_probs, k=5, labels=np.arange(len(label)))

    plt.figure(figsize=(12, 10))
    plt.title(
        f'Top 1 Accuracy : {overallAccuracy*100:3.2f}% | Top 5 Accuracy : {top_5_accuracy*100:3.2f}% ', size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues', fmt='g')

    plt.show()
    print(f'Top 1 Accuracy: {overallAccuracy*100:3.3f}%')
    print(f'Top 5 Accuracy: {top_5_accuracy*100}%')
    print(f'Classwise Accuracy Score: \n{classwiseAccuracy}')

    #print(metrics.classification_report(Y_test, Y_pred))

# %%


def list_to_nparray(list_of_tensors):
    return np.array([x[0].numpy() for x in list_of_tensors])

# %% [markdown]
# ## 2. Plot the epoch vs. loss curve for training and validation data. Save your best model after appropriate hyperparameter tuning. [5]

# %% [markdown]
# #### Trying with different Optimizers
#
# 1. Adam


# %%
criterion = nn.NLLLoss()
learning_rate = 0.005
n_hidden = 128
n_linear = 128
rnn = RNN(n_letters, n_hidden, n_linear, n_categories)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
loss_list = model_training(rnn, criterion, optimizer,
                           x_train, y_train, x_test, y_test, num_epochs=10)
plot_training_graphs(loss_list)


with torch.no_grad():

    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(rnn.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)

# %% [markdown]
# 2. Adagrad

# %%
criterion = nn.NLLLoss()
learning_rate = 0.005
n_hidden = 128
n_linear = 128
rnn = RNN(n_letters, n_hidden, n_linear, n_categories)
optimizer = torch.optim.Adagrad(rnn.parameters(), lr=learning_rate)
loss_list = model_training(rnn, criterion, optimizer,
                           x_train, y_train, x_test, y_test, num_epochs=10)
plot_training_graphs(loss_list)


with torch.no_grad():

    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(rnn.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)

# %% [markdown]
# 3. SGD

# %%
criterion = nn.NLLLoss()
learning_rate = 0.005
n_hidden = 128
n_linear = 128
rnn = RNN(n_letters, n_hidden, n_linear, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
loss_list = model_training(rnn, criterion, optimizer,
                           x_train, y_train, x_test, y_test, num_epochs=20)
plot_training_graphs(loss_list)


with torch.no_grad():

    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(rnn.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)


# %% [markdown]
# #### Trying with different hidden layer and linear layer size

# %%
criterion = nn.NLLLoss()
learning_rate = 0.005

hidden_sizes = [512, 256, 128, 64]
linear_sizes = [512, 256, 128, 64]

best_score = 0.0
best_model = None
for n_hidden in hidden_sizes:
    for n_linear in linear_sizes:
        print(f'Trying n_hidden={n_hidden} and n_linear={n_linear} :\n')
        rnn = RNN(n_letters, n_hidden, n_linear, n_categories)
        optimizer = torch.optim.Adagrad(rnn.parameters(), lr=learning_rate)
        loss_list = model_training(
            rnn, criterion, optimizer, x_train, y_train, x_test, y_test, num_epochs=15)
        plot_training_graphs(loss_list)
        with torch.no_grad():
            test_output = []
            for i, line_tensor in enumerate(x_test):
                test_output.append(rnn.evaluate(line_tensor))
            np_y_test = list_to_nparray(y_test)
            np_test_output = list_to_nparray(test_output)

            confusionMatrixAndAccuracyReport(
                np_y_test, np_test_output, all_categories)
            Y_pred = np_test_output.argmax(axis=1)
            score = metrics.f1_score(np_y_test, Y_pred, average='weighted')
            print(
                f'\n\t With n_hidden={n_hidden} and n_linear={n_linear} : \t f1_score={score:.3f}\t accuracy={metrics.accuracy_score(np_y_test,Y_pred):.3f}')
            if score > best_score:
                best_score = score
                best_model = rnn

print(f'\nBest model = {best_model}\t with f1 score = {best_score}\n')

# %%
optimizer = torch.optim.Adagrad(best_model.parameters(), lr=learning_rate)
loss_list = model_training(best_model, criterion, optimizer,
                           x_train, y_train, x_test, y_test, num_epochs=10)
plot_training_graphs(loss_list)

# %% [markdown]
# ## 3. Obtain a Confusion Matrix on validation data for your best model. [2]

# %%
with torch.no_grad():

    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(best_model.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)

# %% [markdown]
# ## 4. Add three more linear layers to our current RNN architecture and perform 2 & 3 again.[4]

# %%


class RNNExtended(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, output_size):
        super(RNNExtended, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size

        self.i2lh = nn.Linear(input_size + hidden_size, linear_size)

        # New three layers
        self.lh1 = nn.Linear(linear_size, linear_size)
        self.lh2 = nn.Linear(linear_size, linear_size)
        self.lh3 = nn.Linear(linear_size, linear_size)

        self.l2h = nn.Linear(linear_size, hidden_size)

        self.i2lo = nn.Linear(input_size + hidden_size, linear_size)

        # New three layers
        self.lo1 = nn.Linear(linear_size, linear_size)
        self.lo2 = nn.Linear(linear_size, linear_size)
        self.lo3 = nn.Linear(linear_size, linear_size)

        self.l2o = nn.Linear(linear_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2lh(combined)
        hidden = self.tanh(hidden)
        # New three layers
        hidden = self.lh1(hidden)
        hidden = self.tanh(hidden)
        hidden = self.lh1(hidden)
        hidden = self.tanh(hidden)
        hidden = self.lh1(hidden)
        hidden = self.tanh(hidden)

        hidden = self.l2h(hidden)
        hidden = self.tanh(hidden)

        output = self.i2lo(combined)
        output = self.tanh(output)
        # New three layers
        output = self.lo1(output)
        output = self.tanh(output)
        output = self.lo2(output)
        output = self.tanh(output)
        output = self.lo3(output)
        output = self.tanh(output)

        output = self.l2o(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def evaluate(self, line_tensor):
        hidden = self.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

        return output

    def predict(self, input_line, n_predictions=3):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = self.evaluate(lineToTensor(input_line))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])


# %%
criterion = nn.NLLLoss()
learning_rate = 0.002
n_hidden = 128
n_linear = 512
rnn_x = RNNExtended(n_letters, n_hidden, n_linear, n_categories)
optimizer = torch.optim.Adagrad(rnn_x.parameters(), lr=learning_rate)
loss_list = model_training(rnn_x, criterion, optimizer,
                           x_train, y_train, x_val, y_val, num_epochs=50)
plot_training_graphs(loss_list)


# %% [markdown]
# ## 5. Report test accuracy for both the above architectures.

# %%

with torch.no_grad():

    print('RNN Model First One')
    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(best_model.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)

    print('RNN Model with extra three linear LAYER')

    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(rnn_x.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)

# %% [markdown]
# ## 6.1 Build a stacked RNN (2 RNN blocks) model and do appropriate hyperparameter tuning.

# %%


class RNNStacked(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, output_size):
        super(RNNStacked, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size

        self.rnn1 = RNN(input_size, n_hidden, n_linear,
                        n_linear, out_softmax=False)
        self.rnn2 = RNN(n_linear, n_hidden, n_linear, output_size)

    def forward(self, input, hidden1, hidden2):
        output, hidden1 = self.rnn1(input, hidden1)
        output, hidden2 = self.rnn2(output, hidden2)
        return output, hidden1, hidden2

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def evaluate(self, line_tensor):
        hidden1 = self.initHidden()
        hidden2 = self.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden1, hidden2 = self(line_tensor[i], hidden1, hidden2)

        return output

    def predict(self, input_line, n_predictions=3):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = self.evaluate(lineToTensor(input_line))

            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])

# %%
# Train the model


def stacked_model_training(rnn_stacked, criterion, optimizer, x_train, y_train, x_val, y_val, num_epochs=10):
    start = time.time()
    loss_list = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        for i, line_tensor in enumerate(x_train):
            category_tensor = y_train[i]
            hidden1 = rnn_stacked.initHidden()
            hidden2 = rnn_stacked.initHidden()
            rnn_stacked.zero_grad()
            # Forward pass
            for i in range(line_tensor.size()[0]):
                outputs, hidden1, hidden2 = rnn_stacked(
                    line_tensor[i], hidden1, hidden2)

            loss = criterion(outputs, category_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += metrics.accuracy_score(
                category_tensor, outputs.argmax(axis=1))
        # Evaluate the model on the validation set
        with torch.no_grad():
            #images, labels = next(iter(testloader))
            for i, line_tensor in enumerate(x_val):
                outputs = rnn_stacked.evaluate(line_tensor)
                loss = criterion(outputs, y_val[i])
                val_loss += loss.item()
                val_acc += metrics.accuracy_score(
                    y_val[i], outputs.argmax(axis=1))
        train_loss = train_loss/len(x_train)
        val_loss = val_loss/len(x_val)
        train_acc = train_acc/len(x_train)
        val_acc = val_acc/len(x_val)
        print(f'Epoch: {epoch+1} ({timeSince(start)}) \tTraining Loss: {train_loss:.3f}, \tTest Loss: {val_loss:.3f},  \tTraining acc: {train_acc:.2f}, \tTest acc: {val_acc:.2f}, ')
        loss_list.append([train_loss, val_loss, train_acc, val_acc])

    print(
        f'Training completed in {timeSince(start)} \tTraining Loss: {loss_list[-1][0]:.3f}, \tTest Loss: {loss_list[-1][1]:.3f},  \tTraining acc: {loss_list[-1][2]:.2f}, \tTest acc: {loss_list[-1][3]:.2f}, ')
    return np.array(loss_list)


# %%
criterion = nn.NLLLoss()
learning_rate = 0.005
n_hidden = 128
n_linear = 256
rnn_stacked = RNNStacked(n_letters, n_hidden, n_linear, n_categories)
optimizer = torch.optim.Adagrad(rnn_stacked.parameters(), lr=learning_rate)
loss_list = stacked_model_training(
    rnn_stacked, criterion, optimizer, x_train, y_train, x_val, y_val, num_epochs=30)
plot_training_graphs(loss_list)
with torch.no_grad():

    test_output = []
    for i, line_tensor in enumerate(x_test):
        test_output.append(rnn_stacked.evaluate(line_tensor))

    confusionMatrixAndAccuracyReport(list_to_nparray(
        y_test), list_to_nparray(test_output), all_categories)

# %% [markdown]
# ## 6.2 At last, perform inference on the following words and print their language of origin.
#
# Emilia, Alexandra, Sachiko, Vladimir, Minh, Xi, Muammar, Mukesh, Andrew, Ronaldo

# %%
test_name_list = ['Emilia', 'Alexandra', 'Sachiko', 'Vladimir',
                  'Minh', 'Xi', 'Muammar', 'Mukesh', 'Andrew', 'Ronaldo']

for name in test_name_list:
    rnn_stacked.predict(name)
