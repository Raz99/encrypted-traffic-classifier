# Celery task + feature engineering for async model inference
import joblib
import pandas as pd
import numpy as np
import io
from celery import Celery

# Setup Celery
celery_app = Celery('tasks', 
                    broker='redis://redis:6379/0', 
                    backend='redis://redis:6379/0')

# Load models and encoders
app_model = joblib.load('models/application_model.pkl')
app_le = joblib.load('models/application_encoder.pkl')
att_model = joblib.load('models/attribution_model.pkl')
att_le = joblib.load('models/attribution_encoder.pkl')

# Feature Engineering for Application
def engineer_features_app(df):
    """Feature engineering for the application model."""
    df = df.copy()

    # Protocol encoding (match training)
    if 'Protocol' in df.columns:
        df['Protocol_num'] = df['Protocol'].map({'tcp': 0, 'udp': 1})

    fwd_len, bwd_len = df['fwd_packets_length'], df['bwd_packets_length']
    fwd_cnt, bwd_cnt = df['fwd_packets_amount'], df['bwd_packets_amount']
    total_pkts = fwd_cnt + bwd_cnt
    total_bytes = fwd_len + bwd_len
    df['total_pps'] = df['pps_fwd'] + df['pps_bwd']
    df['estimated_duration'] = total_pkts / (df['total_pps'] + 1e-6)
    df['avg_iat'] = df['estimated_duration'] / (total_pkts + 1e-6)
    df['avg_packet_size'] = total_bytes / (total_pkts + 1e-6)
    df['avg_fwd_pkt_size'] = fwd_len / (fwd_cnt + 1e-6)
    df['avg_bwd_pkt_size'] = bwd_len / (bwd_cnt + 1e-6)
    df['bytes_per_second'] = total_bytes / (df['estimated_duration'] + 1e-6)
    df['payload_ratio'] = bwd_len / (fwd_len + 1)
    df['packet_ratio'] = bwd_cnt / (fwd_cnt + 1)
    df['log_total_bytes'] = np.log1p(total_bytes)

    first_cols = [f'first_packet_sizes_{i}' for i in range(15)]
    if all(c in df.columns for c in first_cols):
        df['handshake_avg'] = df[first_cols].mean(axis=1)
        df['handshake_std'] = df[first_cols].std(axis=1)
        df['handshake_max'] = df[first_cols].max(axis=1)
    
    df['fwd_bwd_pkts_diff'] = fwd_cnt - bwd_cnt
    df['fwd_bwd_bytes_diff'] = fwd_len - bwd_len

    # Clean infinities and NaNs
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Feature Engineering for Attribution
def engineer_features_att(df):
    df = df.copy()
    
    # Protocol encoding (match training)
    if 'Protocol' in df.columns:
        df['Protocol_num'] = df['Protocol'].map({'tcp': 0, 'udp': 1})

    total_packets = df['fwd_packets_amount'] + df['bwd_packets_amount']
    total_pps = df['pps_fwd'] + df['pps_bwd']

    # Calculate fundamental features
    df['estimated_duration'] = total_packets / (total_pps + 1e-6)
    df['bytes_per_second'] = (df['fwd_packets_length'] + df['bwd_packets_length']) / (df['estimated_duration'] + 1e-6)
    df['bytes_per_packet'] = (df['fwd_packets_length'] + df['bwd_packets_length']) / (total_packets + 1e-6)
    df['payload_ratio'] = df['bwd_packets_length'] / (df['fwd_packets_length'] + 1)

    # Fill silence_ratio if missing (model expects it)
    if 'silence_windows' in df.columns:
        df['silence_ratio'] = df['silence_windows'] / (total_packets + 1)
    else:
        df['silence_ratio'] = 0

    df['burstiness'] = total_packets / (df['estimated_duration'] + 1e-6)
    df['rate_stability'] = df['bytes_per_second'] / (total_packets + 1)
    df['chat_signature'] = df['silence_ratio'] * df['burstiness']
    df['stream_signature'] = df['bytes_per_second'] * (1 - df['silence_ratio'])

    # Clean infinities and NaNs
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

@celery_app.task(name="tasks.run_inference")
def run_inference(data_json, task_type):
    """Run model inference for a JSON-serialized DataFrame."""
    df = pd.read_json(io.StringIO(data_json), orient='split') # JSON to DataFrame
    
    if task_type == 'app':
        # Application task
        processed_df = engineer_features_app(df)
        X = processed_df[app_model.feature_names_in_] # Match training columns
        preds_numeric = app_model.predict(X)
        return app_le.inverse_transform(preds_numeric).tolist()
    
    else:
        # Attribution task
        processed_df = engineer_features_att(df)
        X = processed_df[att_model.feature_names_in_] # Match training columns
        preds_numeric = att_model.predict(X)
        return att_le.inverse_transform(preds_numeric).tolist()