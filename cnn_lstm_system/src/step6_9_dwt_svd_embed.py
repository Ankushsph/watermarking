# Steps 6-9: DWT-LL + SVD-EA + TR-EMB + SYNC-MRK

import numpy as np
import pywt
import os
from .config import *

def dwt_decompose_ll_only(frame):
    """
    Step 6: DWT-LL - Decompose audio using DWT, select only LL subband
    LL subband is the most stable (low frequency approximation)
    """
    coeffs = pywt.wavedec(frame, WAVELET_TYPE, level=DWT_LEVEL)
    ll_subband = coeffs[0]  # LL subband only
    return ll_subband, coeffs

def dwt_reconstruct(coeffs):
    """Reconstruct signal from DWT coefficients"""
    return pywt.waverec(coeffs, WAVELET_TYPE)

def compute_teo_weighted_alpha(frame, is_voiced, teo_value):
    """
    Step 7: SVD-EA - TEO-weighted adaptive alpha
    
    Voiced frames with high TEO -> High alpha (strong embedding)
    Unvoiced frames -> Medium alpha
    """
    # Normalize TEO value
    teo_normalized = np.clip(teo_value / 1000.0, 0, 1)
    
    if is_voiced:
        # Voiced: High alpha based on TEO
        alpha = ALPHA_VOICED_MEDIUM + (ALPHA_VOICED_HIGH - ALPHA_VOICED_MEDIUM) * teo_normalized
    else:
        # Unvoiced: Medium alpha
        alpha = ALPHA_UNVOICED_MEDIUM
    
    return alpha

def svd_embed_in_block(ll_block, bit, alpha):
    """
    Step 7: SVD-EA - Apply SVD and embed bit with adaptive alpha
    
    Args:
        ll_block: 4x4 block from LL subband
        bit: Bit to embed (0 or 1)
        alpha: Embedding strength
    
    Returns:
        modified_block: Block with embedded watermark
        original_s0: Original largest singular value (for reference)
    """
    # Ensure 4x4 block
    if ll_block.shape != (SVD_BLOCK_SIZE, SVD_BLOCK_SIZE):
        ll_block = ll_block[:SVD_BLOCK_SIZE, :SVD_BLOCK_SIZE]
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(ll_block, full_matrices=False)
    
    original_s0 = S[0]
    
    # Embed bit by modifying largest singular value
    if bit == 1:
        S[0] = S[0] + alpha
    else:
        S[0] = S[0] - alpha
    
    # Reconstruct block
    modified_block = np.dot(U, np.dot(np.diag(S), Vt))
    
    return modified_block, original_s0

def triple_redundancy_embed(ll_subband, bit, alpha):
    """
    Step 8: TR-EMB - Triple Redundancy Embedding
    Embed same watermark 3 times in 3 different bands (Low, Mid, High)
    
    Args:
        ll_subband: LL subband coefficients
        bit: Bit to embed
        alpha: Embedding strength
    
    Returns:
        modified_ll: LL subband with triple redundancy embedding
        original_values: Original S[0] values from all 3 bands
    """
    modified_ll = ll_subband.copy()
    original_values = []
    
    # Ensure we have enough coefficients
    min_required = BAND_HIGH_RANGE[1]
    if len(modified_ll) < min_required:
        modified_ll = np.pad(modified_ll, (0, min_required - len(modified_ll)), mode='edge')
    
    # Band 1: Low (0-16)
    low_block = modified_ll[BAND_LOW_RANGE[0]:BAND_LOW_RANGE[1]].reshape(SVD_BLOCK_SIZE, SVD_BLOCK_SIZE)
    low_modified, low_orig = svd_embed_in_block(low_block, bit, alpha)
    modified_ll[BAND_LOW_RANGE[0]:BAND_LOW_RANGE[1]] = low_modified.flatten()
    original_values.append(low_orig)
    
    # Band 2: Mid (16-32)
    mid_block = modified_ll[BAND_MID_RANGE[0]:BAND_MID_RANGE[1]].reshape(SVD_BLOCK_SIZE, SVD_BLOCK_SIZE)
    mid_modified, mid_orig = svd_embed_in_block(mid_block, bit, alpha)
    modified_ll[BAND_MID_RANGE[0]:BAND_MID_RANGE[1]] = mid_modified.flatten()
    original_values.append(mid_orig)
    
    # Band 3: High (32-48)
    high_block = modified_ll[BAND_HIGH_RANGE[0]:BAND_HIGH_RANGE[1]].reshape(SVD_BLOCK_SIZE, SVD_BLOCK_SIZE)
    high_modified, high_orig = svd_embed_in_block(high_block, bit, alpha)
    modified_ll[BAND_HIGH_RANGE[0]:BAND_HIGH_RANGE[1]] = high_modified.flatten()
    original_values.append(high_orig)
    
    return modified_ll, original_values

def add_sync_markers(watermark_bits):
    """
    Step 9: SYNC-MRK - Insert sync markers at regular intervals
    Format: [SYNC][Block1][SYNC][Block2][SYNC][Block3]
    
    Args:
        watermark_bits: Watermark bits to protect
    
    Returns:
        marked_bits: Bits with sync markers inserted
    """
    if not USE_SYNC_MARKERS:
        return watermark_bits
    
    marked_bits = []
    sync_pattern = SYNC_PATTERN
    
    # Add initial sync marker
    marked_bits.extend(sync_pattern)
    
    # Insert sync markers at intervals
    for i in range(0, len(watermark_bits), SYNC_INTERVAL):
        block = watermark_bits[i:i+SYNC_INTERVAL]
        marked_bits.extend(block)
        marked_bits.extend(sync_pattern)
    
    return np.array(marked_bits, dtype=int)

def detect_sync_markers(received_bits):
    """
    Step 9: SYNC-MRK - Detect and remove sync markers with fuzzy matching
    Uses Hamming distance for robustness to bit flips
    
    Args:
        received_bits: Received bits with sync markers
    
    Returns:
        watermark_bits: Bits without sync markers
        sync_positions: Detected sync marker positions
    """
    if not USE_SYNC_MARKERS:
        return received_bits, []
    
    sync_pattern = np.array(SYNC_PATTERN)
    sync_len = len(sync_pattern)
    sync_positions = []
    watermark_bits = []
    
    i = 0
    while i < len(received_bits):
        # Check if we have enough bits left for a sync marker
        if i + sync_len <= len(received_bits):
            segment = received_bits[i:i+sync_len]
            matches = np.sum(segment == sync_pattern)
            
            # Fuzzy matching: 62.5% threshold (5/8 bits) - balanced between robustness and false positives
            if matches >= 5:  # 5 out of 8 bits must match
                sync_positions.append(i)
                i += sync_len  # Skip the sync marker
                continue
        
        # Not a sync marker, keep the bit
        if i < len(received_bits):
            watermark_bits.append(received_bits[i])
        i += 1
    
    return np.array(watermark_bits, dtype=int), sync_positions

def embed_watermark_with_cnn_lstm(frames, voiced_indices, unvoiced_indices, 
                                   watermark_bits, teo_values, cnn_lstm_model=None):
    """
    Complete embedding pipeline: Steps 6-9
    DWT-LL + SVD-EA + TR-EMB + SYNC-MRK
    
    Args:
        frames: Audio frames
        voiced_indices: Indices of voiced frames
        unvoiced_indices: Indices of unvoiced frames
        watermark_bits: Bits to embed (after BCH encoding and spreading)
        teo_values: TEO values for each frame
        cnn_lstm_model: Trained CNN-LSTM model (optional)
    
    Returns:
        watermarked_frames: Frames with embedded watermark
        reference_values: Original singular values for extraction
    """
    # Step 9: Add sync markers
    marked_bits = add_sync_markers(watermark_bits)
    
    watermarked_frames = frames.copy()
    reference_values = []
    bit_index = 0
    
    # Prioritize voiced frames for embedding
    embedding_indices = voiced_indices + unvoiced_indices
    
    # Store embedding indices for stable extraction
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/embedding_indices.npy', np.array(embedding_indices))
    
    for frame_idx in embedding_indices:
        if bit_index >= len(marked_bits):
            break
        
        frame = frames[frame_idx]
        is_voiced = frame_idx in voiced_indices
        teo_value = teo_values[frame_idx] if frame_idx < len(teo_values) else 0
        
        # Step 6: DWT decomposition (LL only)
        ll_subband, coeffs = dwt_decompose_ll_only(frame)
        
        # Step 7: Compute TEO-weighted adaptive alpha
        alpha = compute_teo_weighted_alpha(frame, is_voiced, teo_value)
        
        # If CNN-LSTM model available, use it to refine alpha
        if cnn_lstm_model is not None:
            # TODO: Use CNN-LSTM to predict optimal embedding strength
            pass
        
        # Step 8: Triple redundancy embedding
        if USE_TRIPLE_REDUNDANCY and len(ll_subband) >= BAND_HIGH_RANGE[1]:
            modified_ll, orig_values = triple_redundancy_embed(
                ll_subband, marked_bits[bit_index], alpha
            )
            # Store with frame index for proper alignment
            reference_values.append((frame_idx, orig_values))
        else:
            # Fallback: single embedding
            if len(ll_subband) >= 16:
                block = ll_subband[:16].reshape(4, 4)
                modified_block, orig_s0 = svd_embed_in_block(block, marked_bits[bit_index], alpha)
                modified_ll = ll_subband.copy()
                modified_ll[:16] = modified_block.flatten()
                reference_values.append((frame_idx, [orig_s0]))
            else:
                modified_ll = ll_subband
                reference_values.append((frame_idx, [0]))
        
        # Reconstruct frame
        coeffs[0] = modified_ll
        try:
            watermarked_frame = dwt_reconstruct(coeffs)
            if len(watermarked_frame) > len(frame):
                watermarked_frame = watermarked_frame[:len(frame)]
            elif len(watermarked_frame) < len(frame):
                watermarked_frame = np.pad(watermarked_frame, 
                                          (0, len(frame) - len(watermarked_frame)), 
                                          mode='edge')
            watermarked_frames[frame_idx] = watermarked_frame
        except:
            watermarked_frames[frame_idx] = frame
        
        bit_index += 1
    
    print(f"DWT-SVD Embedding Complete:")
    print(f"  Embedded {bit_index} bits in {len(embedding_indices)} frames")
    print(f"  Triple redundancy: {USE_TRIPLE_REDUNDANCY}")
    print(f"  Sync markers: {USE_SYNC_MARKERS}")
    
    # Save reference values with frame indices for proper alignment
    np.save('outputs/reference_values.npy', np.array(reference_values, dtype=object))
    
    return watermarked_frames, reference_values
