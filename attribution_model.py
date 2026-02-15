# Train and export the attribution-classification model
import gc
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Feature Engineering
def engineer_features(df):
    df = df.copy()

    # Protocol encoding
    if 'Protocol' in df.columns:
        df['Protocol_num'] = df['Protocol'].map({'tcp': 0, 'udp': 1})

    total_packets = df['fwd_packets_amount'] + df['bwd_packets_amount'] # Calculate total packets
    total_pps = df['pps_fwd'] + df['pps_bwd'] # Calculate total packets per second

    # 1e-6 to avoid division by zero
    df['estimated_duration'] = total_packets / (total_pps + 1e-6) # Estimated duration of the flow
    df['bytes_per_second'] = (df['fwd_packets_length'] + df['bwd_packets_length']) / (df['estimated_duration'] + 1e-6) # Throughput in bytes per second
    df['bytes_per_packet'] = (df['fwd_packets_length'] + df['bwd_packets_length']) / (total_packets + 1e-6) # Average bytes per packet

    df['payload_ratio'] = df['bwd_packets_length'] / (df['fwd_packets_length'] + 1) # Ratio of backward to forward payload

    if 'silence_windows' in df.columns:
        df['silence_ratio'] = df['silence_windows'] / (total_packets + 1) # Ratio of silence windows to total packets

    df['burstiness'] = total_packets / (df['estimated_duration'] + 1e-6)
    df['rate_stability'] = df['bytes_per_second'] / (total_packets + 1)
    df['chat_signature'] = df['silence_ratio'] * df['burstiness']
    df['stream_signature'] = df['bytes_per_second'] * (1 - df['silence_ratio'])

    # Keep model inputs finite + numeric-friendly
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle infinities and NaNs (replace with 0)
    return df


def main():
    print("Loading data...")
    try:
        # Load only attribution files
        train_df = pd.read_csv("data/att/radcom_att_train.csv")
        test_df = pd.read_csv("data/att/radcom_att_test.csv")
    except FileNotFoundError:
        print("Attribution data files not found. Please ensure 'radcom_att_train.csv' and 'radcom_att_test.csv' are present.")
        return

    # Feature Engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Data Cleaning
    drop_cols = [
        'Source_IP', 'Destination_IP', 'Protocol', 'Timestamp',
        'Source_port', 'Destination_port', 'attribution'
    ]

    # Split features and labels
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['attribution']

    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['attribution']

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Free memory
    del train_df, test_df
    gc.collect()

    print(f"Training on {X_train.shape[1]} features...")

    # Feature Selection
    print("Selecting stable features...")
    rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf_selector.fit(X_train, y_train_enc)

    importances = pd.Series(rf_selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    print("Top 25 Most Important Features:")
    # Print the first 25 features and their importance scores
    print(importances.head(25).to_string())

    # Keep only top 25 features
    top_features = importances.head(25).index.tolist()
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    print(f"\nUsing {len(top_features)} features")

    # Final Attribution Model
    model = Pipeline([
        ("scaler", StandardScaler()), # Normalization
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train_enc)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(model, "models/attribution_model.pkl")
    joblib.dump(le, "models/attribution_encoder.pkl") # Save encoder to translate numbers back to labels

    preds = model.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)

    print("\n==============================")
    print(f"Attribution Accuracy: {acc:.4f}")
    print("==============================")

    print("\nClassification Report:")
    print(classification_report(y_test_enc, preds, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test_enc, preds)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=le.classes_.tolist(),
        yticklabels=le.classes_.tolist()
    )
    ax.xaxis.set_ticks_position('top') # Move the labels to top
    ax.xaxis.set_label_position('top') # Move the "Predicted" title to top

    # Ensure results directory exists
    os.makedirs("results/att", exist_ok=True)

    plt.xticks(rotation=45, ha='left')
    plt.title("Attribution Confusion Matrix", pad=20)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/att/att_confusion_matrix.png")

    print("Confusion matrix saved to results/att/att_confusion_matrix.png")

if __name__ == "__main__":
    main()