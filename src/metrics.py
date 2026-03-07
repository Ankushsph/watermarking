# Performance Metrics Module

import numpy as np
from pesq import pesq
from .config import *

def calculate_ber(original_bits, extracted_bits):
    """Calculate Bit Error Rate"""
    if len(original_bits) != len(extracted_bits):
        min_len = min(len(original_bits), len(extracted_bits))
        original_bits = original_bits[:min_len]
        extracted_bits = extracted_bits[:min_len]
    
    errors = np.sum(original_bits != extracted_bits)
    ber = errors / len(original_bits)
    return ber

def calculate_nc(original_bits, extracted_bits):
    """Calculate Normalized Correlation"""
    if len(original_bits) != len(extracted_bits):
        min_len = min(len(original_bits), len(extracted_bits))
        original_bits = original_bits[:min_len]
        extracted_bits = extracted_bits[:min_len]
    
    # Convert to -1, 1 for correlation
    orig = 2 * original_bits - 1
    extr = 2 * extracted_bits - 1
    
    nc = np.sum(orig * extr) / np.sqrt(np.sum(orig ** 2) * np.sum(extr ** 2))
    return nc

def calculate_snr(original, watermarked):
    """Calculate Signal-to-Noise Ratio"""
    noise = watermarked - original
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_pesq_score(original, watermarked):
    """Calculate PESQ score"""
    try:
        score = pesq(SAMPLE_RATE, original, watermarked, 'wb')
        return score
    except:
        return -1.0

def calculate_tamper_detection_rate(true_tampered, detected_tampered):
    """Calculate tamper detection rate"""
    if len(true_tampered) == 0:
        return 1.0 if len(detected_tampered) == 0 else 0.0
    
    correct_detections = len(set(true_tampered) & set(detected_tampered))
    detection_rate = correct_detections / len(true_tampered)
    return detection_rate

def calculate_false_positive_rate(true_authentic, detected_tampered, total_blocks):
    """Calculate false positive rate"""
    false_positives = len(set(detected_tampered) - set(true_authentic))
    authentic_blocks = total_blocks - len(true_authentic)
    
    if authentic_blocks == 0:
        return 0.0
    
    fpr = false_positives / authentic_blocks
    return fpr
