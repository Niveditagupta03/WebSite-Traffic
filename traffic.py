import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the data
data = pd.read_csv('traffic.csv')
traffic = data['traffic'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_traffic = scaler.fit_transform(traffic)

# Split the data into training and testing sets
train_size = int(len(scaled_traffic) * 0.8)
train_data = scaled_traffic[:train_size]
test_data = scaled_traffic[train_size:]

# Create sequences for input-output pairs
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 7  # Length of input sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(1)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
mse = np.mean((predictions - scaler.inverse_transform(y_test))**2)
print('Mean Squared Error:', mse)
