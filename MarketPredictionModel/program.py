import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from datetime import datetime

#Set up stock, time period, data prep, etc
ticker = 'AMZN'
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.dropna()

#Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data =scaler.fit_transform(data.values)

#Create sequences to train data
seq_set_length = 60

def create_sequences(data, seq_length=seq_set_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length, 3]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(scaled_data, seq_set_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

#Create DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

#LTSM Model
class PredictionModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=1, dropout_rate=0.3):
        super(PredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        #Added in a dropout layer to randomly set 0.2 (20%) of nodes to 0
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        #Below code performs dropout
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

model = PredictionModel()

#Training Model
defined_learning_rate = 0.005
criteria = nn.MSELoss()
#Added in a weight_decay parameter to the optimiser which applies a L2 penalty to prevent overfitting.
#This works by adding a penalty to the loss function proportional to the sum of the sqaures of the weights.
optimiser = torch.optim.Adam(model.parameters(), lr=defined_learning_rate, weight_decay=1e-5)

num_trials = 20
losses = []
for epoch in range(num_trials):
    model.train()
    epoch_loss = 0
    for seq, label in train_loader:
        optimiser.zero_grad()
        y_pred = model(seq)
        loss = criteria(y_pred, label)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{num_trials}, Loss: {loss.item():.4f}')

# Results of model
model.eval()
test_data = scaled_data[-seq_set_length:].reshape(1, seq_set_length, 5)
test_data = torch.tensor(test_data, dtype=torch.float32)

predicted_price = model(test_data).item()
predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_price, 0]])[0][3]
print("Predicted Closing Price:", predicted_price)

# Displaying results against actual data
model.eval() 
predictions = []
actual_prices = []

for i in range(len(scaled_data) - seq_set_length):
    seq = scaled_data[i:i + seq_set_length].reshape(1, seq_set_length, 5)
    seq = torch.tensor(seq, dtype=torch.float32)
    
    pred_price = model(seq).item()
    predictions.append(pred_price)
    
    actual_price = scaled_data[i + seq_set_length, 3]
    actual_prices.append(actual_price)

predictions = scaler.inverse_transform([[0, 0, 0, pred, 0] for pred in predictions])
actual_prices = scaler.inverse_transform([[0, 0, 0, act, 0] for act in actual_prices])

# Residuals
mse = mean_squared_error(actual_prices[:, 3], predictions[:, 3])
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices[:, 3], predictions[:, 3])
r2 = r2_score(actual_prices[:, 3], predictions[:, 3])
print(f'Mean Sqaured Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absoloute Error (MAE): {mae:.4f}')
print(f'R-squared (R2): {r2:.4f}')

# Residuals mean and variance calculations
residuals = torch.tensor(actual_prices[:, 3] - predictions[:, 3])
residuals_mean = torch.mean(residuals).item()
residuals_variance = torch.var(residuals).item()
print(f'Residual Mean: {residuals_mean:.4f}')
print(f'Residuals Variance: {residuals_variance:.4f}')

#Predicted vs Actual prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices[:, 3], label='Actual Prices')
plt.plot(predictions[:, 3], label='Predicted Prices')
plt.legend()
plt.xlabel('Days')
plt.ylabel('Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.show()

# Residuals plot
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals (Actual - Predicted Prices)')
plt.xlabel('Days')
plt.ylabel('Residuals')
plt.show()

# Loss curve plot
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()