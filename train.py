import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from model import ParkingLSTM

def create_dataset(df, input_len=8, horizon=4):
    features = ['norm_occ', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day']
    data = df[features].values
    target = df['norm_occ'].values

    X, y = [], []
    for i in range(len(data) - input_len - horizon):
        X.append(data[i : i + input_len])
        y.append(target[i + input_len : i + input_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_and_evaluate():
    df = pd.read_csv("parking_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Feature Engineering
    scaler = MinMaxScaler()
    df['norm_occ'] = scaler.fit_transform(df[['occupied_slots']])
    
    hours = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    days = df['timestamp'].dt.dayofweek
    df['sin_hour'] = np.sin(2 * np.pi * hours / 24)
    df['cos_hour'] = np.cos(2 * np.pi * hours / 24)
    df['sin_day'] = np.sin(2 * np.pi * days / 7)
    df['cos_day'] = np.cos(2 * np.pi * days / 7)

    X, y = create_dataset(df, input_len=8, horizon=4)

    # 80-20 Train-Test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = torch.tensor(X[:split_idx]), torch.tensor(X[split_idx:])
    y_train, y_test = torch.tensor(y[:split_idx]), torch.tensor(y[split_idx:])

    model = ParkingLSTM(input_dim=5, hidden_dim=64, num_layers=2, output_steps=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("Training LSTM Model...")
    for epoch in range(15):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/15], Loss: {loss.item():.5f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()
        targets = y_test.numpy()

    # Inverse transform to original slot counts
    preds_slots = scaler.inverse_transform(preds)
    targets_slots = scaler.inverse_transform(targets)

    mae = np.mean(np.abs(preds_slots - targets_slots))
    rmse = np.sqrt(np.mean((preds_slots - targets_slots) ** 2))
    
    print("\n--- Quantitative Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} slots")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} slots")

    torch.save(model.state_dict(), "parking_lstm.pth")
    print("Model saved as parking_lstm.pth")

if __name__ == "__main__":
    train_and_evaluate()