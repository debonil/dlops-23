# %% [markdown]
# # DLOps Assignment 2: RNN, LSTM and Docker [100 Marks]
#
# ### Submitted By Debonil Ghosh [M21AIE225]

# %% [markdown]
# Q2. The dataset you have been given is Individual household electric power consumption
# dataset. [25]
# (i)Split the dataset into train and test (80:20) and do the basic preprocessing. [10]
# (ii) Use LSTM to predict the global active power while keeping all other important
# features and predict it for the testing days by training the model and plot the real global active
# power and predicted global active power for the testing days and comparing the results. [8]
# (iii) Now split the dataset in train and test (70:30) and predict the global active power for
# the testing days and compare the results with part (ii). [7]

# %%
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
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


# %%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def get_model_mobilenet_cifar10():
    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    model = model.to(device)
    # print(model)
    return model


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Train the model


def model_training(model, criterion, optimizer, trainloader, testloader, num_epochs=10, model_name='model'):
    start = time.time()
    loss_list = []
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
            print(f'Epoch: {epoch+1} ({timeSince(start)}) \tTraining Loss: {train_loss:.3f}, \tTest Loss: {val_loss:.3f},  \tTraining acc: {train_acc:.2f}, \tTest acc: {val_acc:.2f}, ')
            loss_list.append([train_loss, val_loss, train_acc, val_acc])

        print(
            f'Training completed in {timeSince(start)} \tTraining Loss: {loss_list[-1][0]:.3f}, \tTest Loss: {loss_list[-1][1]:.3f},  \tTraining acc: {loss_list[-1][2]:.2f}, \tTest acc: {loss_list[-1][3]:.2f}, ')
        return np.array(loss_list), time.time()-start, loss_list[-1][2], loss_list[-1][3]


# %%
#sns.set(rc={'axes.facecolor': 'lightblue', 'figure.facecolor': 'lightblue'})


def confusionMatrixAndAccuracyReport(Y_test, Y_pred_probs, label):
    Y_pred = Y_pred_probs.argmax(axis=1)
    cm = metrics.confusion_matrix(Y_test, Y_pred)
    overallAccuracy = metrics.accuracy_score(Y_test, Y_pred)

    classwiseAccuracy = cm.diagonal()/cm.sum(axis=1)

    top_5_accuracy = metrics.top_k_accuracy_score(
        Y_test, Y_pred_probs, k=5, labels=np.arange(10))

    plt.figure(figsize=(10, 10))
    plt.title(
        f'Top 1 Accuracy : {overallAccuracy*100:3.2f}% | Top 5 Accuracy : {top_5_accuracy*100:3.2f}% ', size=14)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    sns.heatmap(data=cm, annot=True, square=True,  cmap='Blues', fmt='g')

    plt.show()
    print(f'Top 1 Accuracy: {overallAccuracy*100:3.3f}%')
    print(f'Top 5 Accuracy: {top_5_accuracy*100}%')
    print(f'Classwise Accuracy Score: \n{classwiseAccuracy}')


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
power_series_df = pd.read_csv('../downloads/household_power_consumption.txt', index_col='date', sep=';',
                              na_values=['nan', '?'], infer_datetime_format=True, parse_dates={'date': ['Date', 'Time']}, low_memory=False)

print(power_series_df.shape)
print(power_series_df.describe().to_markdown())


# %%
power_series_df.sample(10)


# %%

i = 1
cols = [0, 1, 3, 4, 5, 6]
plt.figure(figsize=(20, 15))
for col in cols:
    plt.subplot(len(cols), 1, i)
    plt.plot(power_series_df.resample('M').mean().values[:, col])
    plt.title(
        power_series_df.columns[col] + ' data resample over month for mean', y=0.75, loc='left')
    i += 1
plt.show()


# %%
i = 1
cols = [0, 1, 3, 4, 5, 6]
plt.figure(figsize=(20, 10))
for col in cols:
    plt.subplot(len(cols), 1, i)
    plt.plot(power_series_df.resample('D').mean().values[:, col])
    plt.title(
        power_series_df.columns[col] + ' data resample over day for mean', y=0.75, loc='center')
    i += 1
plt.show()


# %%

def reframe_scaled_value(series_data, no_of_inputs, no_of_outputs):
    # print(series_data)
    n_vars = 1 if type(series_data) is list else series_data.shape[1]
    n_vars_arr = [x for x in range(no_of_inputs)]
    series_data_df = pd.DataFrame(series_data)
    print(f'new data famre = {series_data_df}')
    cols, names = list(), list()
    for i in range(no_of_inputs, 0, -1):
        cols.append(series_data_df.shift(-i))
        names.extend([('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)])
    (f'column names = {names[0:0]}')
    for i in range(0, no_of_outputs):
        cols.append(series_data_df.shift(-i))
        if len(n_vars_arr) < 0:
            n_vars_arr.append(cols)
        if i == 0:
            names.extend([('var%d(t)' % (j+1)) for j in range(n_vars)])
        else:
            names.extend([('var%d(t+%d)' % (j+1)) for j in range(n_vars)])
        #print(f'final output = {series_data_df[0:0]}')
        for _ in n_vars_arr:
            _ += 2
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        return agg


# %%
df_resample_raw = power_series_df.resample('h')
df_resample = df_resample_raw.mean()
print(f'df_resample.shape => {df_resample.shape}')


# %%
# Data Preprocessing

values = df_resample.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = reframe_scaled_value(scaled, 1, 1)
r = list(range(df_resample.shape[1]+1, 2*df_resample.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)
reframed.head()


# %%


# %%
def split_train_test_data(values=reframed.values, split_ratio=0.8):
    values = torch.tensor(values).float()
    split_length = math.floor(len(values) * split_ratio)
    train, test = values[:split_length], values[split_length:]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(f'train_x=>{train_x.shape}, test_x=>{test_x.shape}')
    print(f'train_y=>{train_y.shape}, test_y=>{test_y.shape}')
    return train_x, test_x, train_y, test_y


# %%
# Data spliting into train and test data series.80:20
train_x, test_x, train_y, test_y = split_train_test_data(
    values=reframed.values, split_ratio=0.8)


# %%

# Define the model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # simple model with LSTM
        self.lstm = nn.LSTM(input_size, hidden_size,
                            dropout=1e-1, batch_first=True)
        # and one linear layer
        self.fc_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_output, state = self.lstm(x)
        return self.fc_layer(lstm_output[:, -1, :])

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # for preds
            y_pred = self.forward(x)
        return y_pred


# Instantiate the LSTM model
model = LSTMModel(input_size=train_x.shape[2], hidden_size=100, output_size=1)

print(f'{model.train()}')


# %%
def plot_training_graphs(loss_list):
    fig = plt.figure(figsize=(20, 7))
    plot = fig.add_subplot(1, 1, 1)
    plot.set_title("Training vs Validation loss")
    plot.plot(loss_list[:, 0], linestyle='--', label="Training Loss")
    plot.plot(loss_list[:, 1], linestyle='-', label="Validation Loss")
    plot.set_xlabel("Epoch")
    plot.set_ylabel("Loss")
    plot.legend()
    plt.show()


# %%
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# no of epochs
epoch = 200

# Train the model
loss_list = []
for epoch in range(epoch):
    optimizer.zero_grad()
    output = model(train_x)
    train_loss = criterion(output.squeeze(), train_y)
    train_loss.backward()
    optimizer.step()

    # Evaluate the model on the test data every 10 epochs
    with torch.no_grad():
        test_output = model(test_x.float())
        test_loss = criterion(test_output.squeeze(), test_y)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 10}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
    loss_list.append([train_loss.item(), test_loss.item()])


# %%

# Plot the loss history
plot_training_graphs(np.array(loss_list))


# %%
size = df_resample.shape[1]

# Prediction test
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], size))

# invert scaling for prediction
test_x = test_x.double().numpy()
test_y = test_y.double().numpy()
inv_yhat = np.concatenate((yhat, test_x[:, 1-size:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x[:, 1-size:]), axis=1)
test_y
inv_y = scaler.inverse_transform(inv_y)
#
inv_y = inv_y[:, 0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# Mean squared error in Test dataset
mean_squared_error = mean_squared_error(inv_y, inv_yhat)
print(f'Root Mean squared error in Test dataset : {mean_squared_error:.3f}')
# Root Mean squared error in Test dataset

root_mean_squared_error = np.sqrt(mean_squared_error)
print(
    f'Root Mean squared error in Test dataset : {root_mean_squared_error:.3f}')


# %%
print(f'{inv_y.shape}')

print(f'{inv_yhat.shape}')
comp = pd.DataFrame({"Real_Global_Active_power": inv_y,
                    "Predicted_Global_Active_power": inv_yhat})

aa = [x for x in range(100)]
plt.figure(figsize=(25, 10))
plt.plot(aa, inv_y[:100], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat[:100], 'r', label="predicted Global Active Power")
#
plt.ylabel(power_series_df.columns[0])
plt
plt.xlabel('Time step for first 100 hours')
plt.legend(fontsize=15)
plt.show()


# %%
aa = [x for x in range(1000)]
plt.figure(figsize=(25, 10))
plt.plot(aa, inv_y[1000:2000], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat[1000:2000], 'r',
         label="predicted Global Active Power [80:20]")
plt.ylabel(power_series_df.columns[0])
plt
plt.xlabel('Time step for 1000 hours from 1000 to 2000')
#
plt.legend(fontsize=15)
plt.show()


# %%
# Data spliting into train and test data series.70:30
train_x_70, test_x_30, train_y_70, test_y_30 = split_train_test_data(
    values=reframed.values, split_ratio=0.7)


# %%
# Define the loss function and optimizer
# Initialize the model
model_70_30 = LSTMModel(
    input_size=train_x.shape[2], hidden_size=100, output_size=1)

print(model_70_30)
criterion_70_30 = nn.MSELoss()
optimizer_70_30 = optim.Adam(model_70_30.parameters(), lr=0.001)

# no of epochs
epoch = 200

# Train the model
loss_list_70_30 = []
for epoch in range(epoch):
    optimizer_70_30.zero_grad()
    output = model_70_30(train_x_70)
    train_loss = criterion_70_30(output.squeeze(), train_y_70)
    train_loss.backward()
    optimizer_70_30.step()

    # Evaluate the model on the test data every 10 epochs
    with torch.no_grad():
        test_output = model_70_30(test_x_30.float())
        test_loss = criterion_70_30(test_output.squeeze(), test_y_30)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 10}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
    loss_list_70_30.append([train_loss.item(), test_loss.item()])


# %%
# Plot the loss history
plot_training_graphs(np.array(loss_list))


# %%
size = df_resample.shape[1]

# Prediction test
yhat_30 = model.predict(test_x_30)
test_x_30 = test_x_30.reshape((test_x_30.shape[0], size))

# invert scaling for prediction
inv_yhat_new_30 = np.concatenate((yhat_30, test_x_30[:, 1-size:]), axis=1)
inv_yhat_new_30 = scaler.inverse_transform(inv_yhat_new_30)
inv_yhat_new_30 = inv_yhat_new_30[:, 0]

# invert scaling for actual
test_y_30 = test_y_30.reshape((len(test_y_30), 1))
inv_y_30 = np.concatenate((test_y_30, test_x_30[:, 1-size:]), axis=1)
inv_y_30 = scaler.inverse_transform(inv_y_30)
inv_y_30 = inv_y_30[:, 0]

# Mean squared error in Test dataset
mean_squared_error_30 = mean_squared_error(inv_y_30, inv_yhat_new_30)
print(f'Root Mean squared error in Test dataset : {mean_squared_error_30:.3f}')
# Root Mean squared error in Test dataset

root_mean_squared_error_30 = np.sqrt(mean_squared_error_30)
print(
    f'Root Mean squared error in Test dataset : {root_mean_squared_error_30:.3f}')


# %%
aa = [x for x in range(500)]
plt.figure(figsize=(25, 10))
plt.plot(aa, inv_y[:500], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat_new_30[:500], 'r',
         label="predicted Global Active Power[70:30]")
plt.ylabel(power_series_df.columns[0], size=15)
plt.xlabel('Time step for first 500 hours', size=15)
plt.legend(fontsize=15)
plt.show()


# %%
comp['Predicted Global Active Power[70:30]'] = inv_yhat_new_30[0:6832]
aa = [x for x in range(500)]
plt.figure(figsize=(25, 10))
plt.plot(aa, inv_y[5000:5500], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat_new_30[5000:5500], 'r',
         label="predicted Global Active Powern[70:30]")
plt.ylabel(power_series_df.columns[0], size=15)
plt.xlabel('Time step for 500 hours from 5,000 to 5,500', size=15)
plt.legend(fontsize=15)
plt.show()


# %%
comp.shape
aa = [x for x in range(100)]
plt.figure(figsize=(25, 10))
plt.plot(aa, inv_y[6700:6800], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat[6700:6800], marker='.',
         label="predicted Global Active Power[80:20]")
plt.__doc__
plt.plot(aa, inv_yhat_new_30[6700:6800], 'r',
         label="predicted Global Active Power (70 - 30)")
print
plt.ylabel(power_series_df.columns[0], size=15)
plt.xlabel('Comparing the two prediction with Real Output', size=15)
plt.legend(fontsize=15)
plt.show()


# %%
comp.sample(20)
