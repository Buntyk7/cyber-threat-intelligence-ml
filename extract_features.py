from scapy.all import IP, TCP, UDP
import pandas as pd

def extract_packet_features(packet):
    features = {
        'packet_length': len(packet),
        'protocol': 0,
        'src_port': 0,
        'dst_port': 0,
        'flags': 0
    }

    if IP in packet:
        proto = packet[IP].proto
        features['protocol'] = proto

        if TCP in packet:
            features['src_port'] = packet[TCP].sport
            features['dst_port'] = packet[TCP].dport
            features['flags'] = int(packet[TCP].flags)
        elif UDP in packet:
            features['src_port'] = packet[UDP].sport
            features['dst_port'] = packet[UDP].dport

    return pd.DataFrame([features])
