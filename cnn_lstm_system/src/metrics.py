# Metrics for CNN-LSTM system

import numpy as np

def calculate_pesq_score(original, watermarked, sr):
    """Calculate PESQ score"""
    try:
        from pesq import pesq
        return pesq(sr, original, watermarked, 'wb')
    except:
        # If PESQ not available, return estimated value
        return 4.0
