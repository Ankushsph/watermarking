# Step 1: TEO-FC - Teager Energy Operator Frame Classification

import numpy as np
from scipy.signal import hilbert
from .config import *

def frame_audio(audio, frame_length_samples, overlap=0.5):
    """Split audio into overlapping frames"""
    hop_length = int(frame_length_samples * (1 - overlap))
    frames = []
    
    for start in range(0, len(audio) - frame_length_samples + 1, hop_length):
        frame = audio[start:start + frame_length_samples]
        frames.append(frame)
    
    return np.array(frames)

def compute_teo(frame):
    """
    Compute Teager Energy Operator
    TEO[x(n)] = x²(n) - x(n-1) * x(n+1)
    """
    teo = np.zeros(len(frame))
    for n in range(1, len(frame) - 1):
        teo[n] = frame[n]**2 - frame[n-1] * frame[n+1]
    
    return np.mean(np.abs(teo))

def compute_zcr(frame):
    """Compute Zero Crossing Rate"""
    signs = np.sign(frame)
    signs[signs == 0] = -1
    zcr = np.sum(np.abs(np.diff(signs))) / (2 * len(frame))
    return zcr

def compute_ste(frame):
    """Compute Short-Time Energy"""
    return np.sum(frame ** 2) / len(frame)

def classify_frames_teo(audio, sample_rate=SAMPLE_RATE):
    """
    Step 1: TEO-FC - Classify frames into voiced/unvoiced using TEO
    
    Returns:
        frames: numpy array of frames
        voiced_indices: list of voiced frame indices
        unvoiced_indices: list of unvoiced frame indices
    """
    frame_length_samples = int(FRAME_LENGTH_MS * sample_rate / 1000)
    frames = frame_audio(audio, frame_length_samples, FRAME_OVERLAP)
    
    teo_values = []
    zcr_values = []
    ste_values = []
    
    # Compute features for all frames
    for frame in frames:
        teo_values.append(compute_teo(frame))
        zcr_values.append(compute_zcr(frame))
        ste_values.append(compute_ste(frame))
    
    teo_values = np.array(teo_values)
    zcr_values = np.array(zcr_values)
    ste_values = np.array(ste_values)
    
    # Normalize features
    teo_norm = (teo_values - np.mean(teo_values)) / (np.std(teo_values) + 1e-8)
    zcr_norm = (zcr_values - np.mean(zcr_values)) / (np.std(zcr_values) + 1e-8)
    ste_norm = (ste_values - np.mean(ste_values)) / (np.std(ste_values) + 1e-8)
    
    # TEO-based classification
    # Voiced: High TEO, Low ZCR, High STE
    # Unvoiced: Low TEO, High ZCR, Low STE
    
    teo_threshold = np.median(teo_norm)
    zcr_threshold = np.median(zcr_norm)
    ste_threshold = np.median(ste_norm)
    
    voiced_indices = []
    unvoiced_indices = []
    
    for i in range(len(frames)):
        # Voiced frame criteria
        if teo_norm[i] > teo_threshold and zcr_norm[i] < zcr_threshold and ste_norm[i] > ste_threshold:
            voiced_indices.append(i)
        else:
            unvoiced_indices.append(i)
    
    print(f"TEO-FC Classification:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Voiced frames: {len(voiced_indices)} ({100*len(voiced_indices)/len(frames):.1f}%)")
    print(f"  Unvoiced frames: {len(unvoiced_indices)} ({100*len(unvoiced_indices)/len(frames):.1f}%)")
    
    return frames, voiced_indices, unvoiced_indices, teo_values, zcr_values, ste_values
