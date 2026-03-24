# Watermark Extraction with CNN-LSTM

import numpy as np
import pywt
import os
from .config import *
from .step6_9_dwt_svd_embed import dwt_decompose_ll_only, detect_sync_markers
from .step3_ss_mod import spread_spectrum_demodulate
from .step2_rs_bch_ecc import rs_bch_decode

def extract_from_triple_redundancy(ll_subband, reference_values=None):
    """
    Extract bit from triple redundancy bands using majority voting
    Uses RELATIVE comparison: check if S[0] increased or decreased from reference
    
    Args:
        ll_subband: LL subband coefficients
        reference_values: Original S[0] values [low, mid, high] BEFORE embedding
    
    Returns:
        extracted_bit: Recovered bit (0 or 1)
    """
    if len(ll_subband) < BAND_HIGH_RANGE[1]:
        return 0
    
    bits = []
    
    # Extract from Band 1: Low
    low_block = ll_subband[BAND_LOW_RANGE[0]:BAND_LOW_RANGE[1]].reshape(4, 4)
    U, S, Vt = np.linalg.svd(low_block, full_matrices=False)
    
    # KEY CHANGE: Use RELATIVE comparison - did S[0] increase (bit=1) or decrease (bit=0)?
    if reference_values is not None and len(reference_values) > 0:
        # Bit 1 was embedded by ADDING alpha, bit 0 by SUBTRACTING alpha
        # So if current S[0] > reference, it's likely bit 1
        bit_low = 1 if S[0] > reference_values[0] else 0
    else:
        # Adaptive fallback: compare against mean of other singular values
        if len(S) > 1:
            threshold = np.mean(S[1:]) + 1.5 * np.std(S[1:])
            bit_low = 1 if S[0] > threshold else 0
        else:
            bit_low = 1 if S[0] > np.median(ll_subband) else 0
    bits.append(bit_low)
    
    # Extract from Band 2: Mid
    mid_block = ll_subband[BAND_MID_RANGE[0]:BAND_MID_RANGE[1]].reshape(4, 4)
    U, S, Vt = np.linalg.svd(mid_block, full_matrices=False)
    
    if reference_values is not None and len(reference_values) > 1:
        bit_mid = 1 if S[0] > reference_values[1] else 0
    else:
        if len(S) > 1:
            threshold = np.mean(S[1:]) + 1.5 * np.std(S[1:])
            bit_mid = 1 if S[0] > threshold else 0
        else:
            bit_mid = 1 if S[0] > np.median(ll_subband) else 0
    bits.append(bit_mid)
    
    # Extract from Band 3: High
    high_block = ll_subband[BAND_HIGH_RANGE[0]:BAND_HIGH_RANGE[1]].reshape(4, 4)
    U, S, Vt = np.linalg.svd(high_block, full_matrices=False)
    
    if reference_values is not None and len(reference_values) > 2:
        bit_high = 1 if S[0] > reference_values[2] else 0
    else:
        if len(S) > 1:
            threshold = np.mean(S[1:]) + 1.5 * np.std(S[1:])
            bit_high = 1 if S[0] > threshold else 0
        else:
            bit_high = 1 if S[0] > np.median(ll_subband) else 0
    bits.append(bit_high)
    
    # Majority voting (2 out of 3)
    vote_sum = sum(bits)
    extracted_bit = 1 if vote_sum >= 2 else 0
    
    return extracted_bit

def extract_watermark_with_cnn_lstm(frames, voiced_indices, unvoiced_indices, 
                                     num_bits, cnn_lstm_model=None):
    """
    Complete extraction pipeline with CNN-LSTM
    Uses CNN-LSTM model directly for robust bit extraction
    
    Args:
        frames: Watermarked audio frames
        voiced_indices: Indices of voiced frames
        unvoiced_indices: Indices of unvoiced frames
        num_bits: Expected number of bits (with sync markers)
        cnn_lstm_model: Trained CNN-LSTM model (optional)
    
    Returns:
        extracted_metadata: Recovered patient metadata
        extracted_bits: Raw extracted bits
    """
    # Load reference values if available
    reference_values_dict = {}
    if os.path.exists('outputs/reference_values.npy'):
        try:
            ref_data = np.load('outputs/reference_values.npy', allow_pickle=True)
            # Convert to dictionary for frame-indexed lookup
            for item in ref_data:
                if len(item) == 2:
                    frame_idx, ref_vals = item
                    reference_values_dict[frame_idx] = ref_vals
            print(f"[DIAGNOSTIC] Loaded {len(reference_values_dict)} reference values")
        except Exception as e:
            print(f"[DIAGNOSTIC] Failed to load reference values: {e}")
    
    # Try to load embedding indices for stable frame alignment
    embedding_indices = None
    if os.path.exists('outputs/embedding_indices.npy'):
        try:
            embedding_indices = np.load('outputs/embedding_indices.npy', allow_pickle=True)
            print(f"[DIAGNOSTIC] Using embedded frame indices ({len(embedding_indices)} frames)")
        except:
            print(f"[DIAGNOSTIC] Failed to load embedding indices, using TEO classification")
    
    # Use embedded indices if available, otherwise fall back to TEO
    if embedding_indices is not None:
        extraction_indices = embedding_indices
    else:
        extraction_indices = voiced_indices + unvoiced_indices
        print(f"[DIAGNOSTIC] Using TEO classification ({len(extraction_indices)} frames)")
    
    extracted_bits = []
    
    # Use CNN-LSTM model if available for direct bit prediction
    if cnn_lstm_model is not None and hasattr(cnn_lstm_model, 'predict'):
        print(f"[DIAGNOSTIC] Using CNN-LSTM model for direct bit extraction")
        import torch
        
        # Prepare frame sequences for CNN-LSTM
        for i in range(0, len(extraction_indices) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            if len(extracted_bits) >= num_bits:
                break
            
            frame_seq_indices = extraction_indices[i:i+SEQUENCE_LENGTH]
            frame_seq = np.array([frames[idx] for idx in frame_seq_indices if idx < len(frames)])
            
            if len(frame_seq) == SEQUENCE_LENGTH:
                try:
                    # Use CNN-LSTM to predict bit
                    frame_tensor = torch.FloatTensor(frame_seq).unsqueeze(0)
                    if torch.cuda.is_available():
                        frame_tensor = frame_tensor.cuda()
                        cnn_lstm_model = cnn_lstm_model.cuda()
                    
                    with torch.no_grad():
                        output = cnn_lstm_model(frame_tensor)
                        predicted_bit = 1 if output.item() > 0.5 else 0
                        extracted_bits.append(predicted_bit)
                except:
                    # Fallback to SVD if CNN-LSTM fails
                    pass
    
    # If CNN-LSTM didn't extract enough bits, use SVD extraction
    if len(extracted_bits) < num_bits:
        print(f"[DIAGNOSTIC] CNN-LSTM extracted {len(extracted_bits)} bits, using SVD for remaining")
        
        for i, frame_idx in enumerate(extraction_indices):
            if len(extracted_bits) >= num_bits:
                break
            
            if frame_idx >= len(frames):
                continue
            
            frame = frames[frame_idx]
            
            # DWT decomposition
            ll_subband, _ = dwt_decompose_ll_only(frame)
            
            # Get reference for this frame using frame index
            ref_vals = reference_values_dict.get(frame_idx, None)
            
            # Extract bit with triple redundancy
            if USE_TRIPLE_REDUNDANCY:
                bit = extract_from_triple_redundancy(ll_subband, ref_vals)
            else:
                # Simple extraction
                if len(ll_subband) >= 16:
                    block = ll_subband[:16].reshape(4, 4)
                    U, S, Vt = np.linalg.svd(block, full_matrices=False)
                    if ref_vals is not None:
                        bit = 1 if S[0] > ref_vals[0] else 0
                    else:
                        # Adaptive fallback
                        if len(S) > 1:
                            threshold = np.mean(S[1:]) + 2 * np.std(S[1:])
                            bit = 1 if S[0] > threshold else 0
                        else:
                            bit = 1 if S[0] > np.median(ll_subband) else 0
                else:
                    bit = 0
            
            extracted_bits.append(bit)
    
    extracted_bits = np.array(extracted_bits[:num_bits], dtype=int)
    
    print(f"Extraction Complete:")
    print(f"  Extracted {len(extracted_bits)} bits")
    
    # Remove sync markers with fuzzy matching
    watermark_bits, sync_positions = detect_sync_markers(extracted_bits)
    print(f"  Detected {len(sync_positions)} sync markers")
    print(f"  Watermark bits after sync removal: {len(watermark_bits)}")
    
    # Despread (if spread spectrum was used)
    if USE_SPREAD_SPECTRUM:
        # Note: Need PN sequences from embedding - simplified here
        despread_bits = watermark_bits  # Simplified
    else:
        despread_bits = watermark_bits
    
    # BCH decode
    extracted_metadata, decoded_bits = rs_bch_decode(despread_bits)
    
    return extracted_metadata, decoded_bits

def calculate_ber(original_bits, extracted_bits):
    """Calculate Bit Error Rate"""
    min_len = min(len(original_bits), len(extracted_bits))
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
    ber = errors / min_len
    return ber

def calculate_accuracy(original_bits, extracted_bits):
    """Calculate accuracy percentage"""
    return (1 - calculate_ber(original_bits, extracted_bits)) * 100
