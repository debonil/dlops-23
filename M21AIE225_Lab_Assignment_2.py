# %%
import seaborn as sns
from sklearn import metrics
from sklearn.manifold import TSNE
from matplotlib import offsetbox
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# %%
transform = transforms.ToTensor()

traindataset = datasets.FashionMNIST(
    './downloads/', download=True, train=True, transform=transform)
testdataset = datasets.FashionMNIST(
    './downloads/', download=True, train=False, transform=transform)

# %%
bs = 1000
trainloader = torch.utils.data.DataLoader(
    traindataset, batch_size=bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(
    testdataset, batch_size=bs, shuffle=False, num_workers=4)

# %%


def view_samples():

    test_images, labels = next(iter(testloader))

    fig = plt.figure(figsize=(20, 7))
    nrows = 3
    ncols = 10
    b = np.random.randint(0, test_images.shape[0]-nrows*ncols)
    for i in range(nrows*ncols):
        inp = test_images.view(-1, 28, 28)
        plot = fig.add_subplot(nrows, ncols, i+1)
        plot.set_title(testdataset.classes[labels[i+b].cpu().numpy()])
        imgplot = plt.imshow(inp[i+b].cpu(), cmap='gray')
    plt.show()


view_samples()

# %%


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# %%

model = AutoEncoder()

# %%
model

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# %%
epochs = 60
noise_factor = 0.2
loss_list = []
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0

    for images, labels in trainloader:
        # add random noise to the input images
        noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs = model(noisy_imgs.view(-1, 784))
        # calculate the loss
        # the "target" is still the original, not-noisy images
        loss = criterion(outputs, images.view(-1, 784))
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
    with torch.no_grad():
        for images, labels in testloader:
            # add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            # forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = model(noisy_imgs.view(-1, 784))
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion(outputs, images.view(-1, 784))
            # update running training loss
            val_loss += loss.item()*images.size(0)

    # print avg training statistics
    train_loss = train_loss/len(trainloader)
    val_loss = val_loss/len(testloader)
    print(
        f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}, \tTest Loss: {val_loss:.6f}, ')
    loss_list.append([train_loss, val_loss])
loss_list = np.array(loss_list)

# %%
plt.title("Training vs Validation loss")
plt.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
plt.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
test_images, test_labels = next(iter(testloader))

noisy_test_imgs = np.clip(test_images + noise_factor *
                          torch.randn(*test_images.shape), 0., 1.)

test_output = model(noisy_test_imgs.view(-1, 784))


# %% [markdown]
# ### Some original,noise imposed and reconstructed images from above DAE

# %%

fig = plt.figure(figsize=(20, 7))
nrows = 10
b = np.random.randint(0, test_images.shape[0]-nrows)
for i in range(nrows):
    plot = fig.add_subplot(3, nrows, 0*nrows+i+1)
    plot.set_title('Original Image')
    imgplot = plt.imshow(test_images.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

    plot = fig.add_subplot(3, nrows,  1*nrows+i+1)
    plot.set_title('Noisy Image')
    imgplot = plt.imshow(noisy_test_imgs.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

    plot = fig.add_subplot(3, nrows,  2*nrows+i+1)
    plot.set_title('Reconstructed')
    imgplot = plt.imshow(test_output.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

plt.show()

# %%

# Function to Scale and visualize the embedding vectors


def plot_embedding(X_embed, y, title=None):
    with torch.no_grad():
        X = TSNE(n_components=2, learning_rate='auto',
                 init='random', perplexity=40).fit_transform(X_embed)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 20e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(test_images[i].reshape(
                    28, 28), cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# %%
test_embed = model.encoder(test_images.view(-1, 784))

# %%

with torch.no_grad():
    plot_embedding(test_embed, test_labels.numpy())

# %% [markdown]
# ###  1 FC layer with sigmoid activation for 10 class classification

# %% [markdown]
# ##### Roll = M21AIE225
# ##### X = 4, as last digit of roll no. is odd
# ##### Y = sigmoid, as last digit of roll no. is odd

# %%


class SingleFCLayerClassifier(nn.Module):
    def __init__(self):
        super(SingleFCLayerClassifier, self).__init__()
        self.encoder = model.encoder

        self.fc_layer = nn.Sequential(
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_layer(x)
        return x


# %%
classifier_1fc = SingleFCLayerClassifier()
print(classifier_1fc)

# %%
optimizer_1fc = torch.optim.Adam(classifier_1fc.parameters(), lr=0.001)
criterion_1fc = nn.CrossEntropyLoss()

# %%
epochs = 20

loss_list = []
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0

    for images, labels in trainloader:

        # clear the gradients of all optimized variables
        optimizer_1fc.zero_grad()
        # forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs = classifier_1fc(images.view(-1, 784))
        # calculate the loss
        # the "target" is still the original, not-noisy images
        loss = criterion_1fc(outputs, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_1fc.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
    with torch.no_grad():
        for images, labels in testloader:

            # forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = classifier_1fc(images.view(-1, 784))
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion_1fc(outputs, labels)
            # update running training loss
            val_loss += loss.item()*images.size(0)
    # print avg training statistics
    train_loss = train_loss/len(trainloader)
    val_loss = val_loss/len(testloader)
    print(
        f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}, \tTest Loss: {val_loss:.6f}, ')
    loss_list.append([train_loss, val_loss])
loss_list = np.array(loss_list)

# %%
plt.title("Training vs Validation loss for Model with 1FC Layer")
plt.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
plt.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
test_output_1fc = classifier_1fc(test_images.view(-1, 784))

# %%
#sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred, label):
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = np.trace(cm)/sum(cm.flatten())

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    plt.figure(figsize=(10, 10))
    plt.title('Accuracy Score: {0:3.3f}'.format(overallAccuracy), size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues', fmt='g')

    plt.show()
    print('Overall Accuracy Score: {0:3.3f}'.format(overallAccuracy))
    print('Classwise Accuracy Score: {0}'.format(classwiseAccuracy))


# %%
with torch.no_grad():
    confusionMatrixAndAccuracyReport(
        test_labels, test_output_1fc.argmax(axis=1), testdataset.classes)

# %%
test_embed = model.encoder(test_images.view(-1, 784))
with torch.no_grad():
    plot_embedding(test_embed, test_labels.numpy())

# %%
fig = plt.figure(figsize=(20, 7))
nrows = 10
b = np.random.randint(0, test_images.shape[0]-nrows)
for i in range(nrows):
    plot = fig.add_subplot(3, nrows, 0*nrows+i+1)
    plot.set_title('Original Image')
    imgplot = plt.imshow(test_images.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

    plot = fig.add_subplot(3, nrows,  1*nrows+i+1)
    plot.set_title('Noisy Image')
    imgplot = plt.imshow(noisy_test_imgs.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

    plot = fig.add_subplot(3, nrows,  2*nrows+i+1)
    plot.set_title('Reconstructed')
    imgplot = plt.imshow(test_output.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

plt.show()

# %%


class ThreeFCLayerClassifier(nn.Module):
    def __init__(self):
        super(ThreeFCLayerClassifier, self).__init__()
        self.encoder = model.encoder

        self.fc_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_layer(x)
        return x


# %%
classifier_3fc = ThreeFCLayerClassifier()
print(classifier_3fc)

# %%
optimizer_3fc = torch.optim.Adam(classifier_3fc.parameters(), lr=0.001)
criterion_3fc = nn.CrossEntropyLoss()

# %%
epochs = 20
loss_list = []

for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0

    for images, labels in trainloader:

        # clear the gradients of all optimized variables
        optimizer_3fc.zero_grad()
        # forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs = classifier_3fc(images.view(-1, 784))
        # calculate the loss
        # the "target" is still the original, not-noisy images
        loss = criterion_3fc(outputs, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer_3fc.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
    with torch.no_grad():
        for images, labels in testloader:

            # forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = classifier_3fc(images.view(-1, 784))
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion_3fc(outputs, labels)
            # update running training loss
            val_loss += loss.item()*images.size(0)
    # print avg training statistics
    train_loss = train_loss/len(trainloader)
    val_loss = val_loss/len(testloader)
    print(
        f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}, \tTest Loss: {val_loss:.6f}, ')
    loss_list.append([train_loss, val_loss])
loss_list = np.array(loss_list)

# %%
plt.title("Training vs Validation loss for Model with 3FC Layer")
plt.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
plt.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
test_output_3fc = classifier_3fc(test_images.view(-1, 784))

# %%
with torch.no_grad():
    confusionMatrixAndAccuracyReport(
        test_labels, test_output_3fc.argmax(axis=1), testdataset.classes)

# %%
test_embed = model.encoder(test_images.view(-1, 784))
with torch.no_grad():
    plot_embedding(test_embed, test_labels.numpy())

# %%
fig = plt.figure(figsize=(20, 7))
nrows = 10
b = np.random.randint(0, test_images.shape[0]-nrows)
for i in range(nrows):
    plot = fig.add_subplot(3, nrows, 0*nrows+i+1)
    plot.set_title('Original Image')
    imgplot = plt.imshow(test_images.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

    plot = fig.add_subplot(3, nrows,  1*nrows+i+1)
    plot.set_title('Noisy Image')
    imgplot = plt.imshow(noisy_test_imgs.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

    plot = fig.add_subplot(3, nrows,  2*nrows+i+1)
    plot.set_title('Reconstructed Image')
    imgplot = plt.imshow(test_output.view(-1, 28, 28)
                         [b+i].cpu().detach(), cmap='gray')

plt.show()

# %% [markdown]
# Comparison

# %%
print('Classification Report for 1FC Layer')
print(metrics.classification_report(test_labels, test_output_1fc.argmax(axis=1)))
print('Classification Report for 3FC Layer')
print(metrics.classification_report(test_labels, test_output_3fc.argmax(axis=1)))

print(
    f'Accuracy Score for \t 1FC Layer {metrics.accuracy_score(test_labels,test_output_1fc.argmax(axis=1))} \t 3FC Layer {metrics.accuracy_score(test_labels,test_output_3fc.argmax(axis=1))}')
print(f'F1 Score for \t\t 1FC Layer {metrics.f1_score(test_labels,test_output_1fc.argmax(axis=1),average="weighted"):0.3} \t 3FC Layer {metrics.f1_score(test_labels,test_output_3fc.argmax(axis=1),average="weighted"):0.3}')
