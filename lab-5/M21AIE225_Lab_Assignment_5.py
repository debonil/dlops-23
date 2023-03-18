
import math
import time
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# import dataset
from sklearn import datasets
iris = datasets.load_iris()

print(iris.data.shape)
print(iris.target.shape)


# train test split dataset


x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, shuffle=True, random_state=42)


x_train_t = torch.from_numpy(x_train).to(torch.float32)
x_test_t = torch.from_numpy(x_test).to(torch.float32)
y_train_t = torch.from_numpy(y_train).to(torch.long)
y_test_t = torch.from_numpy(y_test).to(torch.long)


x_train_t.shape


#sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred_probs):
    Y_pred = Y_pred_probs.argmax(axis=1)
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = metrics.accuracy_score(Y_test, Y_pred)

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    plt.figure(figsize=(10, 10))
    plt.title(f'Accuracy : {overallAccuracy*100:3.2f}% ', size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues', fmt='g')

    plt.show()
    print(f'Accuracy: {overallAccuracy*100:3.3f}%')
    print(f'Classwise Accuracy Score: \n{classwiseAccuracy}')


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


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Train the model


def model_training(model, criterion, optimizer, num_epochs=10, patience=5):
    start = time.time()
    loss_list = []
    best_val_loss = float('inf')
    num_epochs_since_improvement = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        # Forward pass
        outputs = model(x_train_t)
        loss = criterion(outputs, y_train_t)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += metrics.accuracy_score(y_train_t, outputs.argmax(axis=1))
        # Evaluate the model on the validation set
        with torch.no_grad():
            outputs = model(x_test_t)
            loss = criterion(outputs, y_test_t)
            val_loss += loss.item()
            val_acc += metrics.accuracy_score(y_test_t, outputs.argmax(axis=1))
        loss_list.append([train_loss, val_loss, train_acc, val_acc])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            num_epochs_since_improvement = 0
        else:
            num_epochs_since_improvement += 1

        print(f'Epoch: {epoch+1} ({timeSince(start)}) \tTraining Loss: {train_loss:.3f}, \tTest Loss: {val_loss:.3f},  \tTraining acc: {train_acc:.2f}, \tTest acc: {val_acc:.2f}, \t num_epochs_since_improvement = {num_epochs_since_improvement} ')
        if num_epochs_since_improvement == patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print(
        f'Training completed in {timeSince(start)} \tTraining Loss: {loss_list[-1][0]:.3f}, \tTest Loss: {loss_list[-1][1]:.3f},  \tTraining acc: {loss_list[-1][2]:.2f}, \tTest acc: {loss_list[-1][3]:.2f}, ')
    return np.array(loss_list), time.time()-start, loss_list[-1][2], loss_list[-1][3]


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


model = NeuralNet()
print(model)
optimizer_inst = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()
loss_list, t, train_a, test_a = model_training(
    model, criterion, optimizer_inst, num_epochs=500, patience=5)
plot_training_graphs(loss_list)


test_output = model(x_test_t)
with torch.no_grad():
    print(f'Confusion Matrix ')
    confusionMatrixAndAccuracyReport(y_test_t, test_output)
