# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Dataset and Problem Definition

# +
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# -

raw_data = pd.read_csv('../../../data/orders_sample_v0.3.csv')
raw_data[:5]

sorted_data = raw_data.sort_values('paid_at').reset_index()
sorted_data['day'] = sorted_data.paid_at.map(lambda x : x[5:10])

naive_data = sorted_data[['total_amount', 'day']]
naive_data = naive_data.groupby(['day']).sum()

plt.figure(figsize=(20,10))
plt.plot(naive_data['total_amount'])
plt.show()

# # Data Preprocessing

# +
total_amounut = naive_data.total_amount.values.astype(float)

test_data_size = 1

train_data = total_amounut[:-test_data_size]
test_data = total_amounut[-test_data_size:]
del test_data_size

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)


# -

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


train_window = 7
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

train_inout_seq[-1]

# # Creating LSTM Model

device = torch.device('cuda')


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# +
epochs = 2000
# epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.to(device))

        single_loss = loss_function(y_pred, labels.to(device))
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
# -

# # Making Predictions

# +
fut_pred = 7

test_inputs = train_data_normalized[-train_window:].tolist()
print(test_inputs)

# +
model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq.to(device)).item())
# -

test_inputs[fut_pred:]

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
print(actual_predictions)

x = np.arange(132, 144, 1)
print(x)

x = []
for i in range(7):
    x.append('04-0{}'.format(i+1))
x = np.array(x)

# +
plt.figure(figsize=(20,10))

plt.title('Month vs Total')
plt.ylabel('Total Amount')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(naive_data['total_amount'])
plt.plot(x,actual_predictions)
plt.show()
# -

actual_predictions

submission = pd.read_csv('../../../data/submission_v0.1.csv')

submission.Predicted = actual_predictions

submission.to_csv('./submission_v0.1.csv', index=False)

test2 = pd.read_csv('./submission_v0.1.csv')

test2


