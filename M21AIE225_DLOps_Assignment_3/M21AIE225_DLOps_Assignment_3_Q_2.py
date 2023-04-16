# %% [markdown]
# ## DL Ops Assignment 3
# ### Question 2
# #### Submitted by - Debonil Ghosh [ M21AIE225 ]

# %%
# %%
import os
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from sklearn import metrics
import seaborn as sns
import time
import math
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Available device ==> {device}')


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# %%
!rm - rf ./logs
# Train the model


def model_training(model, criterion, optimizer, trainloader, testloader, num_epochs=10, model_name='model'):
    start = time.time()
    loss_list = []
    model.train()
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './logs/'+model_name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += metrics.accuracy_score(labels.cpu().detach(
                ).numpy(), outputs.cpu().detach().numpy().argmax(axis=1))
                prof.step()
            # Evaluate the model on the validation set
            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_acc += metrics.accuracy_score(labels.cpu().detach(
                    ).numpy(), outputs.cpu().detach().numpy().argmax(axis=1))
            train_loss = train_loss/len(trainloader)
            val_loss = val_loss/len(testloader)
            train_acc = train_acc/len(trainloader)
            val_acc = val_acc/len(testloader)
            print(f'Epoch: {epoch+1} ({timeSince(epoch_start)}) \tTraining Loss: {train_loss:.3f}, \tTest Loss: {val_loss:.3f},  \tTraining acc: {train_acc:.2f}, \tTest acc: {val_acc:.2f}, ')
            loss_list.append([train_loss, val_loss, train_acc, val_acc])

        print(
            f'Training completed in {timeSince(start)} \tTraining Loss: {loss_list[-1][0]:.3f}, \tTest Loss: {loss_list[-1][1]:.3f},  \tTraining acc: {loss_list[-1][2]:.2f}, \tTest acc: {loss_list[-1][3]:.2f}, ')
        return np.array(loss_list), time.time()-start, loss_list[-1][2], loss_list[-1][3]


# %%
#sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred, classes, title=''):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = metrics.accuracy_score(Y_test, Y_pred)

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    f1_score = metrics.f1_score(Y_test, Y_pred, average='weighted')

    plt.figure(figsize=(10, 10))
    plt.title(
        f'{title} : Accuracy : {overallAccuracy*100:3.2f}% | F1 Score : {f1_score*100:3.2f}% ', size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues',
                fmt='g', xticklabels=classes, yticklabels=classes)

    plt.show()
    plt.savefig(
        f'confusion_mat_{title}_{time.time()}.png', bbox_inches='tight')
    print(f'Accuracy: {overallAccuracy*100:3.3f}%')
    print(f'F1 Score: {f1_score*100:3.3f}%')
    classwiseAccuracy_df = pd.DataFrame(
        data=[classwiseAccuracy], columns=classes)
    print(
        f'\nClasswise Accuracy Score: \n{classwiseAccuracy_df.to_markdown(index=False)}')
    print('\nConfusion Matrix:')
    print(cm.to_markdown())


# %%
def plot_training_graphs(loss_list, title=''):
    fig = plt.figure(figsize=(20, 7))
    plot = fig.add_subplot(1, 2, 1)
    plot.set_title(f"{title} : Training vs Validation loss")
    plot.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
    plot.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Loss")
    plot.legend()
    plot = fig.add_subplot(1, 2, 2)
    plot.set_title(f"{title} : Training vs Validation Accuracy")
    plot.plot(loss_list[:, 2], linestyle='--', label="Training Accuracy")
    plot.plot(loss_list[:, 3], linestyle='-', label="Validation Accuracy")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Accuracy")
    plot.legend()
    plt.show()
    plt.savefig(
        f'training_loss_{title}_{time.time()}.png', bbox_inches='tight')


# %% [markdown]
#  Load and preprocessing CIFAR10 dataset using standard augmentation and normalization techniques [10 Marks]

# %%
# Load and preprocessing CIFAR10 dataset using standard augmentation and
# normalization techniques [10 Marks]
# %%
data_path = '../.data'
transform = T.Compose(
    [T.ToTensor()])
train_set = torchvision.datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True)
test_set = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)


# %% [markdown]
# ## Train the following models for profiling them using during the training step [5 *2 = 10Marks]

# %% [markdown]
# A. Conv -> Conv -> Maxpool (2,2) -> Conv -> Maxpool(2,2) -> Conv -> Maxpool(2,2)
#
# i. You can decide the parameters of convolution layers and activations on your own.
#
# ii. Make sure to keep 4 conv-layers and 3 max-pool layers in the order describes above.

# %%


class CustomImageClassifier(nn.Module):
    def __init__(self):
        super(CustomImageClassifier, self).__init__()
        # batch_size = 32
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=256*4*4, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256*4*4)
        x = self.fc_layers(x)
        return x


custom_cnn_model = CustomImageClassifier().to(device)
print(custom_cnn_model.train())


# %%
def train_custom_cnn_model():
    model_name = 'custom_cnn_model'
    optimizer = torch.optim.Adam(custom_cnn_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss_list, t, train_a, test_a = model_training(
        custom_cnn_model, criterion, optimizer, trainloader, testloader, num_epochs=10, model_name=model_name)
    plot_training_graphs(loss_list, title=model_name)
    custom_cnn_model.eval()
    with torch.no_grad():
        test_labels = []
        test_output = []
        for batch in testloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = custom_cnn_model(x)
            test_labels += y.cpu()
            test_output += torch.argmax(y_hat, dim=1).cpu()

        test_labels = np.array(test_labels)
        test_output = np.array(test_output)
        print(f'\nModel Evaluation Summary:')
        confusionMatrixAndAccuracyReport(
            test_labels, test_output, test_set.classes, title=model_name)


# %%
train_custom_cnn_model()


# %% [markdown]
# ○ VGG16

# %%
vgg16_model = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device)
print(vgg16_model.train())


# %%
def train_vgg16_model():
    model_name = 'vgg16_model'
    optimizer = torch.optim.Adam(vgg16_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    loss_list, t, train_a, test_a = model_training(
        vgg16_model, criterion, optimizer, trainloader, testloader, num_epochs=10, model_name=model_name)
    plot_training_graphs(loss_list, title=model_name)
    vgg16_model.eval()
    with torch.no_grad():
        test_labels = []
        test_output = []
        for batch in testloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = vgg16_model(x)
            test_labels += y.cpu()
            test_output += torch.argmax(y_hat, dim=1).cpu()

        test_labels = np.array(test_labels)
        test_output = np.array(test_output)
        print(f'\nModel Evaluation Summary:')
        confusionMatrixAndAccuracyReport(
            test_labels, test_output, test_set.classes, title=model_name)


# %%
train_vgg16_model()


# %%
%load_ext tensorboard

# %%
# %reload_ext tensorboard

# %%
!lsof - i: 6006

# %%
!kill - 9 10710

# %%
#!rm -rf ./logs

# %%
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
!ls

# %%
!cd logs
%tensorboard - -logdir = './'
