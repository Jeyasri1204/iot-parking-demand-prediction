# 🚗 IoT-Based Parking Demand Prediction using Deep Neural Networks

An end-to-end intelligent parking demand prediction and real-time slot management system using IoT sensor stream simulation, Deep LSTM networks, and modern foundation model baselines (Google TimesFM 2.5 and PatchTST).

---

## 📌 Problem Statement
Parking demand fluctuates dynamically based on time of day, campus schedules, and location. Without accurate short-term forecasting, lot space is mismanaged—leading to traffic congestion during peak hours and underutilized slots during off-peak periods.

---

## 🎯 Key Objectives
- **IoT Data Pipeline:** Ingest node-level slot occupancy states via simulated sensor streams using MQTT/HTTP.
- **Deep Sequence Modeling:** Train multi-step PyTorch LSTM architectures with cyclical time features (Sine/Cosine of Hour and Day).
- **Foundation Model Benchmarking:** Compare lightweight LSTM models against state-of-the-art foundation baselines (**Google TimesFM 2.5** and **PatchTST**).
- **Interactive Operations:** Provide a live Streamlit dashboard with real-time slot controllers, dynamic demand forecast curves, and high-occupancy spillover alerts.

---
## 📊 Performance Benchmark

| Model Architecture | MAE (Slots) | RMSE (Slots) | Inference Latency | Deployment Feasibility |
| :--- | :---: | :---: | :---: | :--- |
| **Custom BiLSTM / DNN (Ours)** | **1.42** | **1.95** | **4.2 ms** | Edge / Microcontroller Feasible |
| **PatchTST Baseline** | 1.25 | 1.64 | 18.5 ms | Edge / Cloud Hybrid |
| **TimesFM 2.5 (Zero-Shot)** | 1.10 | 1.48 | 45.0 ms | Cloud Server Ingestion |

---

## 💻 Tech Stack
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Interactive UI:** Streamlit, Plotly
- **Protocols:** MQTT / HTTP API abstraction

---

## 🚀 How to Run Locally

1. Clone the Repository

        git clone [https://github.com/Jeyasri1204/iot-parking-demand-prediction.git](https://github.com/Jeyasri1204/iot-parking-demand-prediction.git)
        cd iot-parking-demand-prediction

2. Install Dependencies
       pip install numpy pandas torch scikit-learn streamlit plotly

3. Generate Simulated IoT Sensor Logs
       python simulator.py

4. Train the LSTM Model
        python train.py

5. Launch the Streamlit Dashboard
       streamlit run app.py
