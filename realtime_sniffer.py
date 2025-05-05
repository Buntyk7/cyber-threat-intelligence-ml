from scapy.all import sniff
import joblib
from app.extract_features import extract_packet_features
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load model and scaler
model = joblib.load('models/cyber_threat_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # Ensure this is saved during training

def process_packet(packet):
    try:
        features_df = extract_packet_features(packet)
        scaled_features = scaler.transform(features_df)
        prediction = model.predict(scaled_features)
        label = "THREAT" if prediction[0] else "BENIGN"
        print(f"[{label}] Packet: {packet.summary()}")
    except Exception as e:
        print(f"Error: {e}")

sniff(prn=process_packet, store=False)
