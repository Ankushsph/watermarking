# Module 2: Energy-Aware DWT-SVD Robust Watermarking

import numpy as np
import pywt
from .config import *

def logistic_chaotic_encryption(watermark_bits, secret_key):
    """Encrypt watermark using 2D logistic chaotic map"""
    x = secret_key
    encrypted_bits = []
    
    for bit in watermark_bits:
        x = LOGISTIC_R * x * (1 - x)
        chaotic_bit = 1 if x > 0.5 else 0
        encrypted_bits.append(bit ^ chaotic_bit)
    
    return np.array(encrypted_bits, dtype=int), x

def logistic_chaotic_decryption(encrypted_bits, secret_key):
    """Decrypt watermark using same chaotic sequence"""
    return logistic_chaotic_encryption(encrypted_bits, secret_key)[0]

def encode_with_repetition(watermark_bits):
    """
    Encode watermark with repetition coding
    Each bit is repeated REPETITION_FACTOR times
    Standard technique in watermarking research for error correction
    """
    if not USE_REPETITION_CODING:
        return watermark_bits
    
    # Repeat each bit
    repeated_bits = np.repeat(watermark_bits, REPETITION_FACTOR)
    return repeated_bits

def decode_with_repetition(repeated_bits):
    """
    Decode watermark with repetition coding using majority voting
    Corrects errors if less than half of repetitions are wrong
    """
    if not USE_REPETITION_CODING:
        return repeated_bits[:48]
    
    # Reshape to (num_bits, REPETITION_FACTOR)
    num_original_bits = 48
    if len(repeated_bits) < num_original_bits * REPETITION_FACTOR:
        # Pad if needed
        repeated_bits = np.pad(repeated_bits, (0, num_original_bits * REPETITION_FACTOR - len(repeated_bits)))
    
    repeated_bits = repeated_bits[:num_original_bits * REPETITION_FACTOR]
    reshaped = repeated_bits.reshape(num_original_bits, REPETITION_FACTOR)
    
    # Majority voting for each bit
    decoded_bits = []
    for bit_group in reshaped:
        # Count 1s and 0s
        ones = np.sum(bit_group == 1)
        zeros = np.sum(bit_group == 0)
        # Majority wins
        decoded_bits.append(1 if ones > zeros else 0)
    
    return np.array(decoded_bits, dtype=int)

def compute_adaptive_alpha(frame):
    """
    Compute adaptive embedding strength based on frame energy
    Fixed: Better normalization and scaling for alpha range 0.005-0.05
    Multiplied by 3 for stronger watermark while maintaining imperceptibility
    """
    energy = np.sum(frame ** 2) / len(frame)
    
    # Normalize energy to [0, 1] range with better scaling
    # Typical speech energy ranges from 0.001 to 0.5
    normalized_energy = np.clip(energy / 0.5, 0, 1)
    
    # Map to alpha range [0.005, 0.05] with square root for better distribution
    # Square root gives more weight to lower energies (more frames get higher alpha)
    # Multiply by 3 for stronger embedding
    alpha = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * np.sqrt(normalized_energy)) * 3.0
    
    return alpha

def dwt_decompose(frame):
    """Apply DWT decomposition"""
    coeffs = pywt.wavedec(frame, WAVELET_TYPE, level=DWT_LEVEL)
    return coeffs

def dwt_reconstruct(coeffs):
    """Reconstruct signal from DWT coefficients"""
    return pywt.waverec(coeffs, WAVELET_TYPE)

def svd_embed_bit(ll_subband, bit, alpha):
    """Embed one bit into LL subband using SVD"""
    # Store original length
    original_length = len(ll_subband)
    
    # Reshape to 2D if needed
    if len(ll_subband.shape) == 1:
        size = int(np.sqrt(len(ll_subband)))
        if size * size > len(ll_subband):
            size = int(np.floor(np.sqrt(len(ll_subband))))
        ll_subband_2d = ll_subband[:size*size].reshape(size, size)
    else:
        ll_subband_2d = ll_subband
    
    U, S, Vt = np.linalg.svd(ll_subband_2d, full_matrices=False)
    
    # Embed bit in singular values
    if bit == 1:
        S[0] += alpha
    else:
        S[0] -= alpha
    
    # Reconstruct
    modified_ll = np.dot(U, np.dot(np.diag(S), Vt))
    modified_flat = modified_ll.flatten()
    
    # Pad or trim to original length
    if len(modified_flat) < original_length:
        modified_flat = np.pad(modified_flat, (0, original_length - len(modified_flat)), mode='edge')
    else:
        modified_flat = modified_flat[:original_length]
    
    return modified_flat

def embed_watermark_voiced_frames(frames, voiced_indices, watermark_bits, secret_key):
    """
    Embed watermark in voiced frames using adaptive alpha DWT-SVD
    Fixed: Use quantization with dithering for better robustness
    """
    # Apply repetition coding for error correction
    encoded_bits = encode_with_repetition(watermark_bits)
    
    # Encrypt watermark
    encrypted_bits, _ = logistic_chaotic_encryption(encoded_bits, secret_key)
    
    watermarked_frames = frames.copy()
    bit_index = 0
    
    for frame_idx in voiced_indices:
        if bit_index >= len(encrypted_bits):
            break
        
        frame = frames[frame_idx]
        
        # Compute adaptive alpha based on frame energy
        alpha = compute_adaptive_alpha(frame)
        
        # DWT decomposition
        coeffs = dwt_decompose(frame)
        ll_coeffs = coeffs[0]
        
        # SVD embedding using dithered quantization
        if len(ll_coeffs) >= 16:
            ll_block = ll_coeffs[:16].reshape(4, 4)
            U, S, Vt = np.linalg.svd(ll_block, full_matrices=False)
            
            # Store original singular value
            S_original = S[0]
            
            # FIXED: Quantization Index Modulation with proper step size
            # Use 2*alpha as quantization step for better separation
            delta = 2 * alpha
            
            # Quantize to nearest multiple of delta
            k = np.floor(S_original / delta)
            
            # Embed bit by choosing quantization level with dithering
            if encrypted_bits[bit_index] == 1:
                # Bit 1: use upper quantization level (0.75 offset)
                S[0] = (k + 0.75) * delta
            else:
                # Bit 0: use lower quantization level (0.25 offset)
                S[0] = (k + 0.25) * delta
            
            # Reconstruct with modified singular value
            modified_block = np.dot(U, np.dot(np.diag(S), Vt))
            ll_coeffs[:16] = modified_block.flatten()
            coeffs[0] = ll_coeffs
            
            # Reconstruct frame
            try:
                watermarked_frame = dwt_reconstruct(coeffs)
                if len(watermarked_frame) > len(frame):
                    watermarked_frame = watermarked_frame[:len(frame)]
                elif len(watermarked_frame) < len(frame):
                    watermarked_frame = np.pad(watermarked_frame, (0, len(frame) - len(watermarked_frame)), mode='edge')
                watermarked_frames[frame_idx] = watermarked_frame
            except:
                watermarked_frames[frame_idx] = frame
        
        bit_index += 1
    
    return watermarked_frames
