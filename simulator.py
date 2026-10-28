import numpy as np
import pandas as pd

def generate_parking_data(filename="parking_data.csv", days=45, capacity=100):
    date_range = pd.date_range(start="2026-08-01", periods=days * 24 * 4, freq="15min")
    records = []

    for ts in date_range:
        hour = ts.hour + ts.minute / 60.0
        is_weekend = 1 if ts.weekday() >= 5 else 0
        
        # Base campus occupancy probability distribution
        if is_weekend:
            base_ratio = 0.15 + 0.1 * np.sin((hour - 6) * np.pi / 12)
        else:
            if 8.5 <= hour <= 10.5:     # Morning peak arrival
                base_ratio = 0.85 + np.random.uniform(-0.05, 0.10)
            elif 12.5 <= hour <= 14.0:   # Lunch rotation
                base_ratio = 0.70 + np.random.uniform(-0.05, 0.05)
            elif 16.5 <= hour <= 18.5:   # Evening departure
                base_ratio = 0.45 + np.random.uniform(-0.08, 0.05)
            elif 19.0 <= hour or hour <= 7.0: # Night low
                base_ratio = 0.10 + np.random.uniform(0.0, 0.05)
            else:
                base_ratio = 0.60 + np.random.uniform(-0.05, 0.05)

        occupied = int(np.clip(base_ratio * capacity + np.random.normal(0, 2), 0, capacity))
        records.append({
            "timestamp": ts,
            "lot_id": "Campus_Lot_A",
            "occupied_slots": occupied,
            "total_capacity": capacity,
            "is_weekend": is_weekend
        })

    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} records in {filename}")

if __name__ == "__main__":
    generate_parking_data()