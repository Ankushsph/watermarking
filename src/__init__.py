"""Medical Speech Watermarking System - Source Modules"""

# Configuration
from .config import (
    SAMPLE_RATE, AUDIO_DURATION, FRAME_LENGTH_MS, FRAME_OVERLAP, MONO_CHANNEL,
    PATIENT_ID_BITS, DOCTOR_ID_BITS, DATE_STAMP_BITS, DIAGNOSIS_CODE_BITS, HOSPITAL_ID_BITS, TOTAL_WATERMARK_BITS,
    WAVELET_TYPE, DWT_LEVEL, ALPHA_MIN, ALPHA_MAX,
    LOGISTIC_R, LOGISTIC_X0,
    MSLE_BLOCK_DURATION, MSLE_FINE_WINDOW_MS, MSLE_MEDIUM_WINDOW_MS, MSLE_COARSE_WINDOW_MS,
    MSLE_FINE_WEIGHT, MSLE_MEDIUM_WEIGHT, MSLE_COARSE_WEIGHT,
    NUM_HEALTHY_FILES, NUM_DYSARTHRIC_FILES, NUM_MEDICAL_INTENT_FILES, TOTAL_FILES,
    DATA_DIR, PREPROCESSED_DIR, WATERMARKED_DIR, RESULTS_DIR
)

# Utilities
from .utils import (
    load_audio, preprocess_audio, save_audio,
    metadata_to_bits, bits_to_metadata, calculate_snr
)

# Module 1: TEO Classification
from .module1_teo_classification import (
    frame_audio, compute_zcr, compute_ste, compute_teo, classify_frames_teo
)

# Module 2: DWT-SVD Embedding
from .module2_dwt_svd_embedding import (
    logistic_chaotic_encryption, logistic_chaotic_decryption,
    compute_adaptive_alpha, dwt_decompose, dwt_reconstruct,
    svd_embed_bit, embed_watermark_voiced_frames
)

# Module 3: DCT Fragile Watermarking
from .module3_dct_fragile import (
    compute_sha256_hash, dct_transform, idct_transform,
    pair_relationship_embed, embed_fragile_watermark_unvoiced
)

# Module 4: MSLE Verification
from .module4_msle_verification import (
    compute_lyapunov_exponent, compute_multi_scale_lyapunov,
    build_msle_signature, verify_msle_integrity
)

# Extraction
from .extraction import (
    extract_watermark_from_voiced, extract_fragile_from_unvoiced, extract_all_watermarks
)

# Attacks
from .attacks import (
    mp3_compression_attack, awgn_attack, pitch_shift_attack, time_stretch_attack,
    lowpass_filter_attack, resampling_attack, amplitude_scaling_attack,
    tampering_attack, apply_attack
)

# Metrics
from .metrics import (
    calculate_ber, calculate_nc, calculate_snr, calculate_pesq_score,
    calculate_tamper_detection_rate, calculate_false_positive_rate
)

__all__ = [
    # Config
    'SAMPLE_RATE', 'AUDIO_DURATION', 'FRAME_LENGTH_MS', 'FRAME_OVERLAP', 'MONO_CHANNEL',
    'PATIENT_ID_BITS', 'DOCTOR_ID_BITS', 'DATE_STAMP_BITS', 'DIAGNOSIS_CODE_BITS', 'HOSPITAL_ID_BITS', 'TOTAL_WATERMARK_BITS',
    'WAVELET_TYPE', 'DWT_LEVEL', 'ALPHA_MIN', 'ALPHA_MAX',
    'LOGISTIC_R', 'LOGISTIC_X0',
    'MSLE_BLOCK_DURATION', 'MSLE_FINE_WINDOW_MS', 'MSLE_MEDIUM_WINDOW_MS', 'MSLE_COARSE_WINDOW_MS',
    'MSLE_FINE_WEIGHT', 'MSLE_MEDIUM_WEIGHT', 'MSLE_COARSE_WEIGHT',
    'NUM_HEALTHY_FILES', 'NUM_DYSARTHRIC_FILES', 'NUM_MEDICAL_INTENT_FILES', 'TOTAL_FILES',
    'DATA_DIR', 'PREPROCESSED_DIR', 'WATERMARKED_DIR', 'RESULTS_DIR',
    # Utils
    'load_audio', 'preprocess_audio', 'save_audio', 'metadata_to_bits', 'bits_to_metadata', 'calculate_snr',
    # Module 1
    'frame_audio', 'compute_zcr', 'compute_ste', 'compute_teo', 'classify_frames_teo',
    # Module 2
    'logistic_chaotic_encryption', 'logistic_chaotic_decryption', 'compute_adaptive_alpha',
    'dwt_decompose', 'dwt_reconstruct', 'svd_embed_bit', 'embed_watermark_voiced_frames',
    # Module 3
    'compute_sha256_hash', 'dct_transform', 'idct_transform', 'pair_relationship_embed', 'embed_fragile_watermark_unvoiced',
    # Module 4
    'compute_lyapunov_exponent', 'compute_multi_scale_lyapunov', 'build_msle_signature', 'verify_msle_integrity',
    # Extraction
    'extract_watermark_from_voiced', 'extract_fragile_from_unvoiced', 'extract_all_watermarks',
    # Attacks
    'mp3_compression_attack', 'awgn_attack', 'pitch_shift_attack', 'time_stretch_attack',
    'lowpass_filter_attack', 'resampling_attack', 'amplitude_scaling_attack', 'tampering_attack', 'apply_attack',
    # Metrics
    'calculate_ber', 'calculate_nc', 'calculate_pesq_score',
    'calculate_tamper_detection_rate', 'calculate_false_positive_rate'
]

