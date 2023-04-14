# %% [markdown]
# DLOPS Assignment 2 Submitted by Saurav Chowdhury(M21AIE256) 
# reference: https://www.kaggle.com/code/winternguyen/predict-household-electric-power-using-lstm

# %% [markdown]
# The dataset you have been given is Individual household electric power consumption
# dataset. [25]
# (i)Split the dataset into train and test (80:20) and do the basic preprocessing. [10]
# (ii) Use LSTM to predict the global active power while keeping all other important
# features and predict it for the testing days by training the model and plot the real global active
# power and predicted global active power for the testing days and comparing the results. [8]

# %%
import numpy as np
import pandas as pd 

# %%
df = pd.read_csv('household_power_consumption.txt', sep=';', 
                 parse_dates={'date' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='date')

# %%
df

# %%

df.shape

# %%
df.isnull().sum()

# %% [markdown]
# Visualising the data over a month

# %%
import matplotlib.pyplot as plt

i = 1
cols=[0, 1, 3, 4, 5, 6]
plt.figure(figsize=(20, 10))
for col in cols:
    ax = plt.subplot(len(cols), 1, i)
    ax.plot(df.resample('M').mean().values[:, col], color='red')
    ax.set_facecolor('lightyellow')
    ax.set_title(df.columns[col] + ' data resample over month for mean', y=0.75, loc='left')
    i += 1
plt.show()


# %% [markdown]
# Visualising the data for the entire period

# %%
i = 1
cols=[0, 1, 3, 4, 5, 6]
plt.figure(figsize=(20, 10))
for col in cols:
    ax = plt.subplot(len(cols), 1, i)
    ax.plot(df.resample('D').mean().values[:, col], color='red')
    ax.set_title(df.columns[col] + ' data resample over day for mean', y=0.75, loc='center')
    ax.set_facecolor('yellow')
    i += 1
plt.show()


# %% [markdown]
# Heat Map of the Data

# %%
import seaborn as sns

# create a pivot table to prepare the data for the heatmap
pivot = df.pivot_table(index=df.index.year, columns=df.columns[0], values=df.columns[1])

# create the heatmap using seaborn
sns.heatmap(pivot, cmap='YlOrRd')

# show the plot
plt.show()


# %%
df = df[['Global_active_power', 'Global_reactive_power', 'Voltage',
       'Global_intensity', 'Sub_metering_2', 'Sub_metering_1','Sub_metering_3']]

# %%
df

# %% [markdown]
# Converting Time Series data to Supervised Learning Series Data 
# data: The original time series data.
# n_in: The number of lag observations as input (default is 1).
# n_out: The number of observations to predict as output (default is 1).
# dropnan: Whether or not to drop rows with NaN values (default is True).

# %%
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(-i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1)) for j in range(n_vars)]        
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

# %% [markdown]
# Resampling the data to 34589 columns

# %%
df_resample = df.resample('h').mean() 
df_resample.shape

# %% [markdown]
# Preprocessing the data and diving the train and test set into 80:20 Ratio

# %%
from sklearn.preprocessing import MinMaxScaler
import math
values = df_resample.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
r = list(range(df_resample.shape[1]+1, 2*df_resample.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)
reframed.head()

# Data spliting into train and test data series.80:20
values = reframed.values
n_train_time =math.floor(len(values)*0.8)
#n_train_time=4000
train = values[:n_train_time, :]
test = values[n_train_time:, :]
print(train.shape)
print(test.shape)
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# %%
test_x.shape

# %%
train_x.dtype

# %% [markdown]
# Training using PYTORCH LIBRARY

# %%

from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Convert data to PyTorch tensors
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).float()
# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
    def predict(self, x):
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred
# Initialize the model
model = LSTMModel(input_size=train_x.shape[2], hidden_size=100, output_size=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
history = []
test_loss_ar=[]
for epoch in range(250):
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output.squeeze(), train_y)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

  
    with torch.no_grad():
        test_output = model(test_x.float())

        test_loss = criterion(test_output.squeeze(), test_y)
        test_loss_ar.append(test_loss)
        print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

# Plot the loss history
plt.plot(history)
plt.plot(test_loss_ar)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(['train', 'test'], loc='upper right')
plt.show()

size = df_resample.shape[1]

# Prediction test
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], size))
print(test_x.dtype)
print(test_x.shape)
# invert scaling for prediction
test_x = test_x.double().numpy()

test_y= test_y.double().numpy()
inv_yhat = np.concatenate((yhat, test_x[:, 1-size:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x[:, 1-size:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# %% [markdown]
# Preidicting 500 Data Samples and Comparing it with Actual Data

# %%
print(inv_y.shape)
print(inv_yhat.shape)
comp=pd.DataFrame({"Real_Global_Active_power":inv_y,"Predicted_Global_Active_power[80:20]":inv_yhat})

aa=[x for x in range(500)]
plt.figure(figsize=(25,10)) 
plt.plot(aa, inv_y[:500], marker='.', color='blue', label="Real Global Active Power")
plt.plot(aa, inv_yhat[:500], color='red', label="Predicted Global Active Power [80:20]")
plt.ylabel(df.columns[0], size=15)
plt.xlabel('Time step for first 500 hours', size=15)
plt.legend(fontsize=15)
plt.gca().set_facecolor('lightgray')
plt.show()


# %% [markdown]
# The Real and Predicted Gloabl Active Power

# %%
comp

# %% [markdown]
# Comparing Real and Predicted Global Active Power for entire Sample

# %%
import matplotlib.pyplot as plt


x = range(len(comp)) 

plt.figure(figsize=(10, 5))
plt.plot(x, comp['Real_Global_Active_power'], label='Real_Global_Active_power')
plt.plot(x, comp['Predicted_Global_Active_power[80:20]'], label='Predicted_Global_Active_power[80:20]')
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('Real vs. Predicted')
plt.legend()
plt.show()


# %% [markdown]
# Box Plot

# %%
import matplotlib.pyplot as plt

# assuming the DataFrame is named 'df'
data = [comp['Real_Global_Active_power'], comp['Predicted_Global_Active_power[80:20]']]

plt.figure(figsize=(10, 5))
plt.boxplot(data, labels=['Real_Global_Active_power', 'Predicted_Global_Active_power[80:20]'])
plt.ylabel('Y axis label')
plt.title('Real_Global_Active_power vs. Predicted_Global_Active_power[80:20] Boxplot')
plt.show()


# %% [markdown]
# Predicitng Next 1000 Points for Global Active Power

# %%
print(inv_y.shape)
print(inv_yhat.shape)
aa=[x for x in range(1000)]
plt.figure(figsize=(25,10)) 
plt.plot(aa, inv_yhat[1000:2000], color='red', label="Predicted Global Active Power [80:20]")
plt.ylabel(df.columns[0], size=15)
plt.xlabel('Time step for 1000 hours from 1000 to 2000', size=15)
plt.legend(fontsize=15)
plt.gca().set_facecolor('lightgray')
plt.show()


# %% [markdown]
# (iii) Now split the dataset in train and test (70:30) and predict the global active power for
# the testing days and compare the results with part (ii).

# %% [markdown]
# Dividing the dataset into Train and Test set in 70:30 ratio

# %%
from sklearn.preprocessing import MinMaxScaler
import math
values = df_resample.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
r = list(range(df_resample.shape[1]+1, 2*df_resample.shape[1]))
reframed.drop(reframed.columns[r], axis=1, inplace=True)
reframed.head()

# Data spliting into train and test data series. 70:30
values = reframed.values

n_train_time =math.floor(len(values)*0.7)

train = values[:n_train_time, :]
test = values[n_train_time:, :]
print(train.shape)
print(test.shape)
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

# %%
test_x.shape


# %% [markdown]
# Training the data with PYTORCH LIBRARY

# %%

from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Convert data to PyTorch tensors
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y).float()
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y).float()
# Define the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
    def predict(self, x):
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred
# Initialize the model
model = LSTMModel(input_size=train_x.shape[2], hidden_size=100, output_size=1)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
history = []
test_loss_ar=[]
for epoch in range(250):
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output.squeeze(), train_y)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

    # Evaluate the model on the test data every 10 epochs
    
    with torch.no_grad():
        test_output = model(test_x.float())

        test_loss = criterion(test_output.squeeze(), test_y)
        test_loss_ar.append(test_loss)
        print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

# Plot the loss history
plt.plot(history)
plt.plot(test_loss_ar)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(['train', 'test'], loc='upper right')
plt.show()

size = df_resample.shape[1]

# Prediction test
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], size))
print(test_x.dtype)
print(test_x.shape)
# invert scaling for prediction
test_x = test_x.double().numpy()

test_y= test_y.double().numpy()
inv_yhat_new = np.concatenate((yhat, test_x[:, 1-size:]), axis=1)
inv_yhat_new = scaler.inverse_transform(inv_yhat_new)
inv_yhat_new = inv_yhat_new[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x[:, 1-size:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat_new))
print('Test RMSE: %.3f' % rmse)

# %% [markdown]
# Plotting the next 500 points For Global Active Power and comparing with Real Power

# %%
aa=[x for x in range(500)]
plt.figure(figsize=(25,10)) 
plt.plot(aa, inv_y[:500], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat_new[:500], 'r', label="predicted Global Active Power[70:30]")
plt.ylabel(df.columns[0], size=15)
plt.xlabel('Time step for first 500 hours', size=15)
plt.legend(fontsize=15)
plt.gca().set_facecolor('lightgray')
plt.show()

# %% [markdown]
# Comparing the Entire Predicted GAP and REAL GAP(GAP:GLobal active Power)

# %%
import matplotlib.pyplot as plt
comp['Predicted_Global_Active_power[70:30]']= inv_yhat_new[0:6832]

x = range(len(comp))

plt.figure(figsize=(10, 5))
plt.plot(x, comp['Real_Global_Active_power'], label='Real_Global_Active_power')
plt.plot(x, comp['Predicted_Global_Active_power[70:30]'], label='Predicted_Global_Active_power[70:30]')
plt.xlabel('X axis label')
plt.ylabel('Y axis label')
plt.title('Real vs. Predicted')
plt.legend()
plt.show()

# %% [markdown]
# Predicting the Next 1000 Points for Global Active Power

# %%

aa=[x for x in range(1000)]
plt.figure(figsize=(25,10)) 

plt.plot(aa, inv_yhat_new[8000:9000], 'r', label="predicted Global Active Power[70:30]")
plt.ylabel(df.columns[0], size=15)
plt.xlabel('Time step for 1000 hours from 25,000 to 26,000', size=15)
plt.legend(fontsize=15)
plt.gca().set_facecolor('lightgray')
plt.show()

# %% [markdown]
# Comparing Real Global Active Power with Predicted Global Active Power with 1. 80:20 train test ratio 2. 70:30 train test ratio

# %%
comp.shape
aa=[x for x in range(100)]
plt.figure(figsize=(25,10)) 
plt.plot(aa, inv_y[6700:6800], marker='.', label="Real Global Active Power")
plt.plot(aa, inv_yhat[6700:6800], marker='.', label="predicted Global Active Power[80:20]")
plt.plot(aa, inv_yhat_new[6700:6800], 'r', label="predicted Global Active Powern[70:30]")
plt.ylabel(df.columns[0], size=15)
plt.xlabel('Comparing the two prediction with Real Output', size=15)
plt.legend(fontsize=15)
plt.show()

# %% [markdown]
# Box Plot Comapring Real and the two Predicted GAP

# %%
import matplotlib.pyplot as plt


data = [comp['Real_Global_Active_power'], comp['Predicted_Global_Active_power[80:20]'],comp['Predicted_Global_Active_power[70:30]']]

plt.figure(figsize=(10, 5))
plt.boxplot(data, labels=['Real_GAP', 'Predicted_GAP[80:20]','Predicted_GAP[70:30]'])
plt.ylabel('Y axis label')
plt.title('Real_Global_Active_power vs. Predicted_Global_Active_power[80:20] vs Predicted_Global_Active_power[70:30] Boxplot')
plt.show()

# %% [markdown]
# The Real Active Power and Predcited Powers (1 &2)

# %%
comp

# %% [markdown]
# MEan Squared Error TO check which Model is better

# %%
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(comp['Real_Global_Active_power'], comp['Predicted_Global_Active_power[80:20]'])
mse2 = mean_squared_error(comp['Real_Global_Active_power'], comp['Predicted_Global_Active_power[70:30]'])


# %%
print(f"Mean SQuared Value for Predicted Global Active Power[80:20]: {mse1} \n Mean SQuared Value for Predicted Global Active Power[70:30]:{mse2}")
if mse1> mse2:
    print(" Predicted Global Active Power[70:30] gives better performance")
else:
    print("  Predicted Global Active Power[80:20] gives better performance")

# %% [markdown]
# Residual Plots

# %%
import matplotlib.pyplot as plt

# Calculate residuals
residuals = comp['Real_Global_Active_power'] - comp['Predicted_Global_Active_power[80:20]']

# Create residual plot
plt.scatter(comp['Predicted_Global_Active_power[80:20]'], residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted_Global_Active_power')
plt.ylabel('Residuals')
plt.show()


# %%
import matplotlib.pyplot as plt

# Calculate residuals
residuals = comp['Real_Global_Active_power'] - comp['Predicted_Global_Active_power[70:30]']

# Create residual plot
plt.scatter(comp['Predicted_Global_Active_power[70:30]'], residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted_Global_Active_power')
plt.ylabel('Residuals')
plt.show()


