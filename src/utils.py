# Utility Functions for Medical Speech Watermarking

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from .config import *

def load_audio(file_path):
    """Load audio file and return samples and sample rate"""
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    return audio, sr

def preprocess_audio(audio, sr):
    """Preprocess audio: resample, normalize, trim/pad"""
    # Resample to target sample rate
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Normalize amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Trim or pad to fixed duration
    target_length = int(AUDIO_DURATION * SAMPLE_RATE)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    
    return audio

def save_audio(audio, file_path, sr=SAMPLE_RATE):
    """Save audio to file"""
    sf.write(file_path, audio, sr)

def metadata_to_bits(patient_id, doctor_id, date_stamp, diagnosis_code, hospital_id):
    """Convert metadata to 48-bit binary stream"""
    bits = []
    bits.extend([int(b) for b in format(patient_id, f'0{PATIENT_ID_BITS}b')])
    bits.extend([int(b) for b in format(doctor_id, f'0{DOCTOR_ID_BITS}b')])
    bits.extend([int(b) for b in format(date_stamp, f'0{DATE_STAMP_BITS}b')])
    bits.extend([int(b) for b in format(diagnosis_code, f'0{DIAGNOSIS_CODE_BITS}b')])
    bits.extend([int(b) for b in format(hospital_id, f'0{HOSPITAL_ID_BITS}b')])
    result = np.array(bits, dtype=int)
    # Ensure exactly 48 bits
    if len(result) != TOTAL_WATERMARK_BITS:
        raise ValueError(f"Watermark must be exactly {TOTAL_WATERMARK_BITS} bits, got {len(result)}")
    return result

def bits_to_metadata(bits):
    """Convert 48-bit binary stream back to metadata"""
    bits = bits.astype(int)
    patient_id = int(''.join(map(str, bits[:8])), 2)
    doctor_id = int(''.join(map(str, bits[8:16])), 2)
    date_stamp = int(''.join(map(str, bits[16:32])), 2)
    diagnosis_code = int(''.join(map(str, bits[32:40])), 2)
    hospital_id = int(''.join(map(str, bits[40:48])), 2)
    return patient_id, doctor_id, date_stamp, diagnosis_code, hospital_id

def calculate_snr(original, watermarked):
    """Calculate Signal-to-Noise Ratio"""
    noise = watermarked - original
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
