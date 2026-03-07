# Attack Simulation Module

import numpy as np
import librosa
from scipy import signal
from pydub import AudioSegment
import io
import soundfile as sf
from .config import *

def mp3_compression_attack(audio, bitrate='128k'):
    """Simulate MP3 compression attack"""
    # Save to temporary buffer as WAV
    buffer_wav = io.BytesIO()
    sf.write(buffer_wav, audio, SAMPLE_RATE, format='WAV')
    buffer_wav.seek(0)
    
    # Convert to MP3 and back
    audio_segment = AudioSegment.from_wav(buffer_wav)
    buffer_mp3 = io.BytesIO()
    audio_segment.export(buffer_mp3, format='mp3', bitrate=bitrate)
    buffer_mp3.seek(0)
    
    # Convert back to WAV
    audio_segment = AudioSegment.from_mp3(buffer_mp3)
    buffer_wav_out = io.BytesIO()
    audio_segment.export(buffer_wav_out, format='wav')
    buffer_wav_out.seek(0)
    
    attacked_audio, _ = sf.read(buffer_wav_out)
    return attacked_audio[:len(audio)]

def awgn_attack(audio, snr_db=20):
    """Add Additive White Gaussian Noise"""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def pitch_shift_attack(audio, n_steps=2):
    """Pitch shifting attack"""
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)

def time_stretch_attack(audio, rate=1.1):
    """Time stretching attack"""
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    # Trim or pad to original length
    if len(stretched) > len(audio):
        return stretched[:len(audio)]
    else:
        return np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')

def lowpass_filter_attack(audio, cutoff_freq=4000):
    """Low-pass filtering attack"""
    nyquist = SAMPLE_RATE / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(5, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio)

def resampling_attack(audio, target_sr=8000):
    """Resampling attack"""
    # Downsample
    downsampled = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=target_sr)
    # Upsample back
    upsampled = librosa.resample(downsampled, orig_sr=target_sr, target_sr=SAMPLE_RATE)
    return upsampled[:len(audio)]

def amplitude_scaling_attack(audio, scale_factor=0.5):
    """Amplitude scaling attack"""
    return audio * scale_factor

def tampering_attack(audio, start_sec=2.0, duration_sec=1.0):
    """Replace a segment with different audio (tampering simulation)"""
    start_sample = int(start_sec * SAMPLE_RATE)
    duration_samples = int(duration_sec * SAMPLE_RATE)
    
    attacked = audio.copy()
    # Replace with stronger random noise to ensure detection
    attacked[start_sample:start_sample + duration_samples] = np.random.randn(duration_samples) * 0.3
    return attacked

def apply_attack(audio, attack_type, **kwargs):
    """Apply specified attack to audio"""
    attacks = {
        'mp3_128': lambda a: mp3_compression_attack(a, '128k'),
        'mp3_64': lambda a: mp3_compression_attack(a, '64k'),
        'awgn_20': lambda a: awgn_attack(a, 20),
        'awgn_10': lambda a: awgn_attack(a, 10),
        'pitch_shift': lambda a: pitch_shift_attack(a, 2),
        'time_stretch': lambda a: time_stretch_attack(a, 1.1),
        'lowpass': lambda a: lowpass_filter_attack(a, 4000),
        'resampling': lambda a: resampling_attack(a, 8000),
        'amplitude': lambda a: amplitude_scaling_attack(a, 0.5),
        'tampering': lambda a: tampering_attack(a, 2.0, 1.0)
    }
    
    if attack_type in attacks:
        return attacks[attack_type](audio)
    else:
        return audio
