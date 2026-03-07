# Module 3: Pair-Relationship DCT Fragile Watermarking

import numpy as np
from scipy.fftpack import dct, idct
import hashlib
from .config import *

def compute_sha256_hash(audio):
    """Compute SHA-256 hash of audio and return first 64 bits"""
    audio_bytes = audio.tobytes()
    hash_obj = hashlib.sha256(audio_bytes)
    hash_hex = hash_obj.hexdigest()
    hash_binary = bin(int(hash_hex, 16))[2:].zfill(256)
    return np.array([int(b) for b in hash_binary[:64]], dtype=int)

def dct_transform(frame):
    """Apply DCT to frame"""
    return dct(frame, norm='ortho')

def idct_transform(coeffs):
    """Apply inverse DCT"""
    return idct(coeffs, norm='ortho')

def pair_relationship_embed(dct_coeffs, watermark_bits):
    """Embed watermark using pair-relationship in high-frequency DCT coefficients"""
    modified_coeffs = dct_coeffs.copy()
    high_freq_start = len(dct_coeffs) // 2
    high_freq_coeffs = modified_coeffs[high_freq_start:]
    
    bit_index = 0
    for i in range(0, len(high_freq_coeffs) - 1, 2):
        if bit_index >= len(watermark_bits):
            break
        
        bit = watermark_bits[bit_index]
        coeff1 = high_freq_coeffs[i]
        coeff2 = high_freq_coeffs[i + 1]
        
        avg = (coeff1 + coeff2) / 2
        delta = abs(coeff1 - coeff2) / 2 + 0.01
        
        if bit == 0:
            # Force coeff1 > coeff2
            high_freq_coeffs[i] = avg + delta
            high_freq_coeffs[i + 1] = avg - delta
        else:
            # Force coeff1 < coeff2
            high_freq_coeffs[i] = avg - delta
            high_freq_coeffs[i + 1] = avg + delta
        
        bit_index += 1
    
    modified_coeffs[high_freq_start:] = high_freq_coeffs
    return modified_coeffs

def embed_fragile_watermark_unvoiced(frames, unvoiced_indices, intermediate_audio):
    """Embed fragile watermark in unvoiced frames using DCT pair-relationship"""
    # Compute hash of intermediate audio
    fragile_bits = compute_sha256_hash(intermediate_audio)
    
    watermarked_frames = frames.copy()
    bit_index = 0
    
    for frame_idx in unvoiced_indices:
        if bit_index >= len(fragile_bits):
            break
        
        frame = frames[frame_idx]
        
        # DCT transform
        dct_coeffs = dct_transform(frame)
        
        # Embed bits using pair-relationship
        bits_to_embed = fragile_bits[bit_index:min(bit_index + 10, len(fragile_bits))]
        modified_coeffs = pair_relationship_embed(dct_coeffs, bits_to_embed)
        
        # Inverse DCT
        watermarked_frame = idct_transform(modified_coeffs)
        watermarked_frames[frame_idx] = watermarked_frame
        
        bit_index += len(bits_to_embed)
    
    return watermarked_frames
