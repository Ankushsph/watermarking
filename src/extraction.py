# Watermark Extraction Module

import numpy as np
import pywt
from scipy.fftpack import dct
from .module1_teo_classification import classify_frames_teo
from .module2_dwt_svd_embedding import logistic_chaotic_decryption, decode_with_repetition
from .config import *

def extract_watermark_from_voiced(frames, voiced_indices, secret_key, num_bits=48):
    """
    Extract watermark from voiced frames using mid-point decision rule
    Matches dithered quantization embedding (0.25/0.75 offsets)
    """
    extracted_bits = []
    
    # Determine how many bits to extract (with repetition overhead if enabled)
    if USE_REPETITION_CODING:
        total_bits_needed = num_bits * REPETITION_FACTOR
    else:
        total_bits_needed = num_bits
    
    # Extract bits using mid-point decision rule
    for frame_idx in voiced_indices:
        if len(extracted_bits) >= total_bits_needed:
            break
        
        if frame_idx >= len(frames):
            continue
        
        frame = frames[frame_idx]
        coeffs = pywt.wavedec(frame, WAVELET_TYPE, level=DWT_LEVEL)
        ll_coeffs = coeffs[0]
        
        if len(ll_coeffs) >= 16:
            ll_block = ll_coeffs[:16].reshape(4, 4)
            U, S, Vt = np.linalg.svd(ll_block, full_matrices=False)
            
            # Compute adaptive alpha (MUST match embedding exactly)
            energy = np.sum(frame ** 2) / len(frame)
            normalized_energy = np.clip(energy / 0.5, 0, 1)
            alpha = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.sqrt(normalized_energy)) * 3.0
            
            # Quantization step (same as embedding)
            delta = 2 * alpha
            
            # Mid-point decision rule
            # Embedding used: bit=0 → (k+0.25)*delta, bit=1 → (k+0.75)*delta
            # Decision boundary: (k+0.5)*delta
            fractional_part = (S[0] / delta) % 1.0
            
            if fractional_part >= 0.5:
                extracted_bits.append(1)
            else:
                extracted_bits.append(0)
        else:
            extracted_bits.append(0)
    
    # Pad if needed
    extracted_bits = np.array(extracted_bits, dtype=int)
    if len(extracted_bits) < total_bits_needed:
        extracted_bits = np.pad(extracted_bits, (0, total_bits_needed - len(extracted_bits)), mode='constant')
    else:
        extracted_bits = extracted_bits[:total_bits_needed]
    
    # Decrypt using chaotic sequence
    decrypted_bits = logistic_chaotic_decryption(extracted_bits, secret_key)
    
    # Apply repetition decoding with majority voting
    corrected_bits = decode_with_repetition(decrypted_bits)
    
    return corrected_bits

def extract_fragile_from_unvoiced(frames, unvoiced_indices, num_bits=64):
    """Extract fragile watermark from unvoiced frames"""
    extracted_bits = []
    
    for frame_idx in unvoiced_indices:
        if len(extracted_bits) >= num_bits:
            break
        
        frame = frames[frame_idx]
        
        # DCT transform
        dct_coeffs = dct(frame, norm='ortho')
        high_freq_start = len(dct_coeffs) // 2
        high_freq_coeffs = dct_coeffs[high_freq_start:]
        
        # Extract bits from pair relationships
        for i in range(0, len(high_freq_coeffs) - 1, 2):
            if len(extracted_bits) >= num_bits:
                break
            
            coeff1 = high_freq_coeffs[i]
            coeff2 = high_freq_coeffs[i + 1]
            
            if coeff1 > coeff2:
                extracted_bits.append(0)
            else:
                extracted_bits.append(1)
    
    return np.array(extracted_bits[:num_bits], dtype=int)

def extract_all_watermarks(audio, secret_key):
    """Extract both robust and fragile watermarks"""
    # Classify frames
    frames, voiced_indices, unvoiced_indices = classify_frames_teo(audio)
    
    # Extract robust watermark
    robust_watermark = extract_watermark_from_voiced(frames, voiced_indices, secret_key)
    
    # Extract fragile watermark
    fragile_watermark = extract_fragile_from_unvoiced(frames, unvoiced_indices)
    
    return robust_watermark, fragile_watermark
