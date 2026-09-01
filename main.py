# ==============================
# Library Import
# ==============================

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import shap

import joblib

print("Libraries Imported Successfully")
# ============================================
# AirSense AI
# DL Air Quality Forecasting
# ============================================

print("=" * 50)
print(" AirSense AI ")
print(" DL Air Quality Forecasting ")
print("=" * 50)
import pandas as pd

# Load Dataset

df = pd.read_csv("data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv")

print("Dataset Loaded Successfully!")
# Check Dataset Information

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)
# ==============================
# Data Preprocessing
# ==============================

# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Dataset Information

print("\nDataset Information:")
df.info()
# ==============================
# Handle Missing Values
# ==============================

# Replace missing values in numeric columns with median
numeric_columns = df.select_dtypes(include=['number']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Check missing values again

print("\nMissing Values After Handling:")
print(df.isnull().sum())
# Handle remaining missing values in categorical columns

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nFinal Missing Values:")
print(df.isnull().sum())
# ==============================
# Date-Time Processing
# ==============================

# Create datetime column

df['datetime'] = pd.to_datetime(
    df[['year', 'month', 'day', 'hour']]
)

# Sort data according to time

df = df.sort_values('datetime')

# Set datetime as index

df = df.set_index('datetime')

print("Date-Time Processing Completed!")

print(df.head())
# ==============================
# Feature Selection
# ==============================

# Select important features

features = [
    'PM2.5',
    'PM10',
    'SO2',
    'NO2',
    'CO',
    'O3',
    'TEMP',
    'PRES',
    'DEWP',
    'RAIN',
    'WSPM'
]

target = 'PM2.5'


# Create final dataset

df_model = df[features]

print("Feature Selection Completed!")

print(df_model.head())

print("\nSelected Features:")
print(df_model.columns)
# ==============================
# Feature Scaling & Train Test Split
# ==============================

from sklearn.preprocessing import MinMaxScaler

# Separate input and target

X = df_model.drop('PM2.5', axis=1)
y = df_model['PM2.5']


# Scaling

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)


# Time-series split (80% train, 20% test)

train_size = int(len(X_scaled) * 0.8)

X_train = X_scaled[:train_size]
X_test = X_scaled[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


print("Scaling Completed!")

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==============================
# Sequence Data Preparation
# ==============================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Create sequence

def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []

    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y.iloc[i+time_steps])

    return np.array(Xs), np.array(ys)


TIME_STEPS = 24


X_train_lstm, y_train_lstm = create_sequences(
    X_train,
    y_train,
    TIME_STEPS
)

X_test_lstm, y_test_lstm = create_sequences(
    X_test,
    y_test,
    TIME_STEPS
)


print("Sequence Data Prepared!")

print("Train Shape:", X_train_lstm.shape)
print("Test Shape:", X_test_lstm.shape)
# ==============================
# LSTM Model Building
# ==============================

# Build LSTM model

lstm_model = Sequential()

lstm_model.add(
    LSTM(
        units=64,
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    )
)

lstm_model.add(Dropout(0.2))

lstm_model.add(Dense(1))


# Compile model

lstm_model.compile(
    optimizer='adam',
    loss='mse'
)


# Model Summary

lstm_model.summary()


# Train model

history = lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)


print("LSTM Training Completed!")
# ==============================
# LSTM Evaluation
# ==============================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# Prediction

lstm_pred = lstm_model.predict(X_test_lstm)


# Convert shape

lstm_pred = lstm_pred.reshape(-1)


# Evaluation

lstm_mae = mean_absolute_error(
    y_test_lstm,
    lstm_pred
)

lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test_lstm,
        lstm_pred
    )
)

lstm_r2 = r2_score(
    y_test_lstm,
    lstm_pred
)


print("LSTM Evaluation Completed!")

print("MAE:", lstm_mae)
print("RMSE:", lstm_rmse)
print("R2 Score:", lstm_r2)
# ==============================
# RNN Model Building
# ==============================

from tensorflow.keras.layers import SimpleRNN

rnn_model = Sequential()

rnn_model.add(
    SimpleRNN(
        units=64,
        input_shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    )
)

rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(1))

# Compile RNN

rnn_model.compile(
    optimizer='adam',
    loss='mse'
)

# Train RNN

rnn_history = rnn_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

print("RNN Training Completed!")

# ==============================
# RNN Evaluation
# ==============================

rnn_pred = rnn_model.predict(
    X_test_lstm,
    verbose=0
)

rnn_pred = rnn_pred.reshape(-1)

rnn_mae = mean_absolute_error(
    y_test_lstm,
    rnn_pred
)

rnn_rmse = np.sqrt(
    mean_squared_error(
        y_test_lstm,
        rnn_pred
    )
)

rnn_r2 = r2_score(
    y_test_lstm,
    rnn_pred
)

print("RNN Evaluation Completed!")

print("RNN MAE:", rnn_mae)
print("RNN RMSE:", rnn_rmse)
print("RNN R2 Score:", rnn_r2)
# ==============================
# GRU Model Building
# ==============================

from tensorflow.keras.layers import GRU

gru_model = Sequential()

gru_model.add(
    GRU(
        units=64,
        input_shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    )
)

gru_model.add(Dropout(0.2))

gru_model.add(Dense(1))

# Compile GRU

gru_model.compile(
    optimizer='adam',
    loss='mse'
)

# Train GRU

gru_history = gru_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

print("GRU Training Completed!")

# ==============================
# GRU Evaluation
# ==============================

gru_pred = gru_model.predict(
    X_test_lstm,
    verbose=0
)

gru_pred = gru_pred.reshape(-1)

gru_mae = mean_absolute_error(
    y_test_lstm,
    gru_pred
)

gru_rmse = np.sqrt(
    mean_squared_error(
        y_test_lstm,
        gru_pred
    )
)

gru_r2 = r2_score(
    y_test_lstm,
    gru_pred
)

print("GRU Evaluation Completed!")

print("GRU MAE:", gru_mae)
print("GRU RMSE:", gru_rmse)
print("GRU R2 Score:", gru_r2)
# ==============================
# Actual vs Predicted Visualization
# ==============================

import matplotlib.pyplot as plt


plt.figure(figsize=(12,5))

plt.plot(
    y_test_lstm[:200],
    label="Actual PM2.5"
)

plt.plot(
    lstm_pred[:200],
    label="Predicted PM2.5"
)

plt.title("Actual vs Predicted PM2.5")

plt.xlabel("Time")

plt.ylabel("PM2.5")

plt.legend()
import os

os.makedirs("plots", exist_ok=True)

plt.savefig(
    "plots/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

plt.show()
# ==============================
# Training & Validation Loss
# ==============================
import os

os.makedirs("plot", exist_ok=True)

plt.figure(figsize=(10,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(
    "plots/lstm_loss.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()


print("Prediction Graph Generated!")
# ==============================
# RNN Training & Validation Loss
# ==============================

plt.figure(figsize=(10,5))

plt.plot(
    rnn_history.history['loss'],
    label='Training Loss'
)

plt.plot(
    rnn_history.history['val_loss'],
    label='Validation Loss'
)

plt.title('RNN Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/rnn_loss.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("RNN Loss Graph Generated!")
# ==============================
# GRU Training & Validation Loss
# ==============================

plt.figure(figsize=(10,5))

plt.plot(
    gru_history.history['loss'],
    label='Training Loss'
)

plt.plot(
    gru_history.history['val_loss'],
    label='Validation Loss'
)

plt.title('GRU Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/gru_loss.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()
plt.close()

print("GRU Loss Graph Generated!")
# ==============================
# Model Performance Comparison
# ==============================

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

# Model names

models = ['LSTM', 'RNN', 'GRU']

# Performance values

mae_values = [
    lstm_mae,
    rnn_mae,
    gru_mae
]

rmse_values = [
    lstm_rmse,
    rnn_rmse,
    gru_rmse
]

r2_values = [
    lstm_r2,
    rnn_r2,
    gru_r2
]

x = np.arange(len(models))
width = 0.35

# ==============================
# MAE and RMSE Comparison
# ==============================

plt.figure(figsize=(10, 6))

plt.bar(
    x - width/2,
    mae_values,
    width,
    label='MAE'
)

plt.bar(
    x + width/2,
    rmse_values,
    width,
    label='RMSE'
)

plt.xticks(x, models)

plt.xlabel("Deep Learning Models")
plt.ylabel("Error Value")

plt.title("LSTM vs RNN vs GRU Performance Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("LSTM vs RNN vs GRU Performance Comparison Graph Generated!")

# ==============================
# R2 Score Comparison
# ==============================

plt.figure(figsize=(10, 6))

plt.bar(
    models,
    r2_values
)

plt.xlabel("Deep Learning Models")
plt.ylabel("R2 Score")

plt.title("LSTM vs RNN vs GRU R2 Score Comparison")

plt.tight_layout()

plt.savefig(
    "plots/r2_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("R2 Comparison Graph Generated!")
# ==============================
# Save Models
# ==============================

import os
import joblib


# Create model folder if not exists

os.makedirs("models", exist_ok=True)

# ==============================
# Save Deep Learning Models
# ==============================

import os

os.makedirs("models", exist_ok=True)

# Save LSTM Model

lstm_model.save(
    "models/lstm_model.h5"
)

# Save RNN Model

rnn_model.save(
    "models/rnn_model.h5"
)

# Save GRU Model

gru_model.save(
    "models/gru_model.h5"
)

# Save Scaler

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("LSTM, RNN and GRU Models Saved Successfully!")
