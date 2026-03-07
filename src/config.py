# Configuration for Medical Speech Watermarking System

# Audio Parameters
SAMPLE_RATE = 16000
AUDIO_DURATION = 3.0  # seconds
FRAME_LENGTH_MS = 20
FRAME_OVERLAP = 0.5
MONO_CHANNEL = 1

# Watermark Parameters
PATIENT_ID_BITS = 8
DOCTOR_ID_BITS = 8
DATE_STAMP_BITS = 16
DIAGNOSIS_CODE_BITS = 8
HOSPITAL_ID_BITS = 8
TOTAL_WATERMARK_BITS = 48

# Error Correction: Repetition Coding
# Standard technique in watermarking research for achieving BER < 0.05
# Each bit is repeated 5 times, majority voting used for decoding
USE_REPETITION_CODING = True
REPETITION_FACTOR = 5  # Optimal balance between BER and capacity

# DWT-SVD Parameters
WAVELET_TYPE = 'db4'
DWT_LEVEL = 2
ALPHA_MIN = 0.005  # Research specification: small alpha for imperceptibility
ALPHA_MAX = 0.05   # Research specification: maximum alpha for robustness

# Chaotic Encryption Parameters
LOGISTIC_R = 3.9
LOGISTIC_X0 = 0.5

# MSLE Parameters
MSLE_BLOCK_DURATION = 1.0  # seconds
MSLE_FINE_WINDOW_MS = 10
MSLE_MEDIUM_WINDOW_MS = 50
MSLE_COARSE_WINDOW_MS = 100
MSLE_FINE_WEIGHT = 0.5
MSLE_MEDIUM_WEIGHT = 0.3
MSLE_COARSE_WEIGHT = 0.2

# Dataset Parameters
NUM_HEALTHY_FILES = 100
NUM_DYSARTHRIC_FILES = 50
NUM_MEDICAL_INTENT_FILES = 50
TOTAL_FILES = 200

# Paths
DATA_DIR = './data'
PREPROCESSED_DIR = './outputs/preprocessed'
WATERMARKED_DIR = './outputs/watermarked'
RESULTS_DIR = './outputs/results'
