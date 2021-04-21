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

# +
# Dataset and Problem Definition

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

raw_data = pd.read_csv('../../../data/orders_sample_v0.3.csv')
raw_data[:5]

sorted_data = raw_data.sort_values('paid_at').reset_index(drop = True)

sorted_data['day'] = sorted_data.paid_at.map(lambda x : x[5:10])
# sorted_data['time'] = sorted_data.paid_at.map(lambda x : x[11:])
sorted_data['hour'] = sorted_data.paid_at.map(lambda x : x[11:13])
# sorted_data['minute'] = sorted_data.paid_at.map(lambda x : x[14:16])
# sorted_data['second'] = sorted_data.paid_at.map(lambda x : x[17:])
sorted_data['day_hour'] = sorted_data['day']+'-'+sorted_data['hour']

# EDA

hour_dict = {}

for i in range(30):
    idx_start = sorted_data.day.tolist().index('03-{:02d}'.format(i+1))
    idx_end = sorted_data.day.tolist().index('03-{:02d}'.format(i+2))
    hour_dict['{:02d}'.format(i+1)] = sorted_data[idx_start:idx_end]

idx_0401 = sorted_data.day.tolist().index('04-01')
hour_dict['31'] = sorted_data[idx_end:idx_0401]
hour_dict['0401'] = sorted_data[idx_0401:]

plt.figure(figsize=(15,60))
for i in range(1,31):
    plt.subplot(15,2,i)
    plt.title('03-{:02d}'.format(i))
    plt.plot(hour_dict['{:02d}'.format(i)][['total_amount', 'hour']].groupby('hour').sum())
    plt.ylim(0,1300000000)
del i, idx_0401, idx_end, idx_start

amount_data_day_hour = sorted_data[['total_amount','day_hour']].groupby('day_hour').sum()

plt.figure(figsize=(20,10))
plt.title('payment flow per hour')
plt.plot(amount_data_day_hour)
plt.xticks(['03-{:02d}-00'.format(x) for x in range(1,32)], labels=['{:02d}'.format(x) for x in range(1,32)])
plt.show()

amount_data_day = sorted_data[['total_amount','day']].groupby('day').sum()

plt.figure(figsize=(20,10))
plt.grid(True)
plt.title('payment flow per day')
plt.plot(amount_data_day[:-1])
plt.show()

# Data Preprocessing

trash = [x for x in amount_data_day.index if x[:2]!='03']
if len(trash)!=0:
    amount_data_day.drop(trash, inplace = True)
del trash
total_amounut = amount_data_day.total_amount.values.astype(float)

test_data_size = 4

train_data = total_amounut[:-test_data_size]
test_data = total_amounut[-test_data_size:]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

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

# Creating LSTM Model

writer = SummaryWriter('./runs/kakao_commerce')

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

device = torch.device('cuda')
model = LSTM(1,100,1).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# epochs = 2000
epochs = 10000

model.train()
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.to(device))

        single_loss = loss_function(y_pred, labels.to(device))
        single_loss.backward()
        optimizer.step()
        
        writer.add_scalar('Loss/train', single_loss, i)
    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
writer.close()
del i, seq, labels, epochs

# Making Predictions

## for Test set

fut_pred = test_data_size
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()
for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        test_inputs.append(model(seq.to(device)).item())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

x = amount_data_day[-fut_pred:].index.tolist()
x = np.array(x)

plt.figure(figsize=(20,10))

plt.title('Total Amount / day (test set in 03-28 ~ 03-31)')
plt.ylabel('Total Amount')
plt.grid(True)
plt.plot(amount_data_day, label='real')
plt.plot(x,actual_predictions, label='Pred')
plt.legend(fontsize = 20)
plt.show()

## for Future Data

total_data_normalized = scaler.fit_transform(total_amounut .reshape(-1, 1))
total_data_normalized = torch.FloatTensor(total_data_normalized).view(-1)
total_window = 14
total_inout_seq = create_inout_sequences(total_data_normalized, total_window)

fut_pred = 7
total_inputs = total_data_normalized[-total_window:].tolist()

model.eval()
for i in range(fut_pred):
    seq = torch.FloatTensor(total_inputs[-total_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))
        total_inputs.append(model(seq.to(device)).item())

total_actual_predictions = scaler.inverse_transform(np.array(total_inputs[total_window:] ).reshape(-1, 1))

x = ['04-{:02d}'.format(i+1) for i in range(fut_pred)]
x = np.array(x)

plt.figure(figsize=(20,10))

plt.title('Total Amount / day (04-01 ~ 04-07)')
plt.ylabel('Total Amount')
plt.grid(True)
plt.plot(amount_data_day[:-1], label='1')
plt.plot(x,total_actual_predictions, label='pred')
plt.legend(fontsize = 20)
plt.show()

# Submit result

submission = pd.read_csv('../../../data/submission_v0.1.csv')

submission.Predicted = total_actual_predictions

submission.to_csv('./submission.csv', index=False)
