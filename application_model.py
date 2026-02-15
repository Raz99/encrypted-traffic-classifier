# Train and export the application-classification model
import gc
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Advanced Feature Engineering
def engineer_features(df):
    """Add derived features; return a numeric, NaN/inf-safe DataFrame."""
    df = df.copy()

    # Protocol encoding
    if 'Protocol' in df.columns:
        df['Protocol_num'] = df['Protocol'].map({'tcp': 0, 'udp': 1})

    fwd_len, bwd_len = df['fwd_packets_length'], df['bwd_packets_length']
    fwd_cnt, bwd_cnt = df['fwd_packets_amount'], df['bwd_packets_amount']
    total_pkts = fwd_cnt + bwd_cnt
    total_bytes = fwd_len + bwd_len
    df['total_pps'] = df['pps_fwd'] + df['pps_bwd']

    # Time-based features (Inter-Arrival Time)
    df['estimated_duration'] = total_pkts / (df['total_pps'] + 1e-6) # Estimated duration based on volume and rate
    df['avg_iat'] = df['estimated_duration'] / (total_pkts + 1e-6) # Average time between packet arrivals

    # Directional Size features
    df['avg_packet_size'] = total_bytes / (total_pkts + 1e-6)
    df['avg_fwd_pkt_size'] = fwd_len / (fwd_cnt + 1e-6)
    df['avg_bwd_pkt_size'] = bwd_len / (bwd_cnt + 1e-6)
    
    # Ratios and Flow behavior
    df['bytes_per_second'] = total_bytes / (df['estimated_duration'] + 1e-6)
    df['payload_ratio'] = bwd_len / (fwd_len + 1)
    df['packet_ratio'] = bwd_cnt / (fwd_cnt + 1)
    df['log_total_bytes'] = np.log1p(total_bytes)

    # Handshake Signatures
    first_cols = [f'first_packet_sizes_{i}' for i in range(15)]
    if all(c in df.columns for c in first_cols):
        df['handshake_avg'] = df[first_cols].mean(axis=1)
        df['handshake_std'] = df[first_cols].std(axis=1)
        df['handshake_max'] = df[first_cols].max(axis=1)
    
    # Directional asymmetry
    df['fwd_bwd_pkts_diff'] = fwd_cnt - bwd_cnt # Difference in packet counts
    df['fwd_bwd_bytes_diff'] = fwd_len - bwd_len # Difference in byte counts

    # Keep model inputs finite + numeric-friendly
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def main():
    print("Loading data...")
    try:
        # Load only application files
        app_train = pd.read_csv("data/app/radcom_app_train.csv")
        app_test = pd.read_csv("data/app/radcom_app_test.csv")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return

    # Process Physical Features
    train_df = engineer_features(app_train)
    test_df = engineer_features(app_test)

    # Columns to remove from training
    drop_cols = ['Source_IP', 'Destination_IP', 'Protocol', 'Timestamp', 
                 'Source_port', 'Destination_port', 'label'] 
    
    # Define features and labels without stacking probabilities
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    
    y_train = train_df['label']
    y_test = test_df['label']

    # Label Encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Cleanup memory
    del train_df, test_df
    gc.collect()

    print(f"Training on {X_train.shape[1]} features...")

    # Model Training
    model = RandomForestClassifier(n_estimators=200, random_state=44, class_weight="balanced")
    model.fit(X_train, y_train_enc)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save the trained model
    joblib.dump(model, "models/application_model.pkl")
    joblib.dump(le, "models/application_encoder.pkl") # Save encoder to translate numbers back to labels

    # Evaluation
    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)

    print("\n==============================")
    print(f"Application Accuracy: {acc:.4f}")
    print("==============================")
    
    # Feature Importance Visualization
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 10 Most Important Features:")
    print(importances.head(10).to_string())

    # Ensure results directory exists
    os.makedirs("results/app", exist_ok=True)

    # Save Classification Report
    report = classification_report(y_test_enc, preds, target_names=le.classes_, zero_division=0)
    with open("results/app/app_classification_report_full.txt", "w") as f:
        f.write(report)
    print("\nClassification Report saved to results/app/app_classification_report_full.txt")

if __name__ == "__main__":
    main()