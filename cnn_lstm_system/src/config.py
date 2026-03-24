# Configuration for CNN-LSTM Enhanced Medical Speech Watermarking System

# Audio Parameters
SAMPLE_RATE = 16000
AUDIO_DURATION = 3.0
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

# Step 2: RS-BCH Error Correction
USE_BCH_CODING = True
BCH_POLYNOMIAL = 8219
BCH_T = 2

# Step 3: Spread Spectrum Modulation
USE_SPREAD_SPECTRUM = True
SPREADING_FACTOR = 4  # Each bit spread across 4 frames

# Step 4: CNN Feature Extraction
CNN_INPUT_SIZE = 320  # Frame size for CNN
CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

# Step 5: LSTM Temporal Modeling
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
SEQUENCE_LENGTH = 10  # Number of frames in sequence

# Step 6: DWT-LL Subband Selection
WAVELET_TYPE = 'db4'
DWT_LEVEL = 2
USE_LL_ONLY = True  # Most stable subband

# Step 7: SVD Energy-Aware Embedding
ALPHA_VOICED_HIGH = 0.5  # Strong embedding for voiced (increased from 0.08)
ALPHA_VOICED_MEDIUM = 0.3  # Increased from 0.05
ALPHA_UNVOICED_MEDIUM = 0.2  # Increased from 0.03
SVD_BLOCK_SIZE = 4  # 4x4 blocks

# Step 8: Triple Redundancy Embedding
USE_TRIPLE_REDUNDANCY = True
BAND_LOW_RANGE = (0, 16)  # Indices in LL subband
BAND_MID_RANGE = (16, 32)
BAND_HIGH_RANGE = (32, 48)

# Step 9: Synchronization Markers
USE_SYNC_MARKERS = True
SYNC_PATTERN = [1, 0, 1, 0, 1, 0, 1, 0]  # 8-bit sync
SYNC_INTERVAL = 16  # Insert sync every 16 bits

# Step 10: PR-DCT Fragile Watermarking
USE_FRAGILE_WATERMARK = True
DCT_HIGH_FREQ_START = 0.5  # Use high frequency coefficients

# Step 11: MSLE Integrity Verification
MSLE_BLOCK_DURATION = 1.0
MSLE_FINE_WINDOW_MS = 10
MSLE_MEDIUM_WINDOW_MS = 50
MSLE_COARSE_WINDOW_MS = 100
MSLE_FINE_WEIGHT = 0.5
MSLE_MEDIUM_WEIGHT = 0.3
MSLE_COARSE_WEIGHT = 0.2

# CNN-LSTM Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8
USE_GPU = True  # Set to False if no GPU available

# Dataset Parameters
NUM_HEALTHY_FILES = 100
NUM_DYSARTHRIC_FILES = 50
NUM_MEDICAL_INTENT_FILES = 50
TOTAL_FILES = 200

# Paths
DATA_DIR = '../data'
PREPROCESSED_DIR = './outputs/preprocessed'
WATERMARKED_DIR = './outputs/watermarked'
RESULTS_DIR = './outputs/results'
MODEL_DIR = './models'
CNN_MODEL_PATH = './models/cnn_feature_extractor.pth'
LSTM_MODEL_PATH = './models/lstm_temporal_model.pth'
