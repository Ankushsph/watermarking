# Utility functions for CNN-LSTM system

import numpy as np
import soundfile as sf
from .config import SAMPLE_RATE

def load_audio(filepath, target_sr=SAMPLE_RATE):
    """Load audio file and resample if needed"""
    audio, sr = sf.read(filepath)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != target_sr:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_samples)
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio, target_sr

def save_audio(filepath, audio, sr=SAMPLE_RATE):
    """Save audio to file"""
    sf.write(filepath, audio, sr)

def calculate_snr(original, watermarked):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - watermarked) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
