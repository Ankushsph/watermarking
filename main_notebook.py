# %% [markdown]
# # Medical Speech Watermarking System
# ## Adaptive Dual-Layer Medical Speech Watermarking Using Energy-Aware DWT-SVD Embedding
# ### and Multi-Scale Lyapunov Exponent Integrity Verification

# %% [markdown]
# ## PHASE 1: Dataset Preparation

# %% Cell 1: Import Libraries and Setup
import os
import sys
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pywt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for src imports
import pathlib
current_dir = pathlib.Path(__file__).parent.resolve() if '__file__' in globals() else pathlib.Path.cwd()
sys.path.insert(0, str(current_dir))

# Import custom modules
import src.config as config
from src.config import *
import src.utils as utils
from src.utils import *
import src.module1_teo_classification as module1
from src.module1_teo_classification import *
import src.module2_dwt_svd_embedding as module2
from src.module2_dwt_svd_embedding import *
import src.module3_dct_fragile as module3
from src.module3_dct_fragile import *
import src.module4_msle_verification as module4
from src.module4_msle_verification import *
import src.extraction as extraction
from src.extraction import *
import src.attacks as attacks
from src.attacks import *
import src.metrics as metrics
from src.metrics import *

print("All libraries imported successfully!")
print(f"Sample Rate: {SAMPLE_RATE} Hz")
print(f"Audio Duration: {AUDIO_DURATION} seconds")
print(f"Total Watermark Bits: {TOTAL_WATERMARK_BITS}")

# %% Cell 2: Create Directory Structure
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(WATERMARKED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Directory structure created:")
print(f"  - {DATA_DIR}")
print(f"  - {PREPROCESSED_DIR}")
print(f"  - {WATERMARKED_DIR}")
print(f"  - {RESULTS_DIR}")

# %% Cell 3: Check Dataset
print("Checking dataset...")

# Check if data directory has files
if os.path.exists(DATA_DIR):
    # Count WAV files recursively
    wav_count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        wav_count += len([f for f in files if f.endswith(('.wav', '.WAV'))])
    
    if wav_count > 0:
        print(f"Dataset found: {wav_count} WAV files in {DATA_DIR}")
    else:
        print(f"⚠ No WAV files found in {DATA_DIR}")
        print("  Please extract the dataset archives first")
else:
    print(f"✗ {DATA_DIR} directory not found")
    print("  Creating directory and checking for archives...")
    os.makedirs(DATA_DIR, exist_ok=True)

# %% Cell 4: Load and Preprocess Audio Files
print("Loading and preprocessing audio files...")

# Find all audio files
audio_files = []
for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(('.wav', '.WAV')):
            audio_files.append(os.path.join(root, file))

print(f"Found {len(audio_files)} audio files")

# Select subset (adjust based on available files)
selected_files = audio_files[:min(TOTAL_FILES, len(audio_files))]
print(f"Selected {len(selected_files)} files for processing")

# Preprocess and save
preprocessed_files = []
for i, file_path in enumerate(tqdm(selected_files, desc="Preprocessing")):
    try:
        audio, sr = load_audio(file_path)
        audio_preprocessed = preprocess_audio(audio, sr)
        
        output_path = os.path.join(PREPROCESSED_DIR, f"audio_{i:04d}.wav")
        save_audio(audio_preprocessed, output_path)
        preprocessed_files.append(output_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print(f"Preprocessed {len(preprocessed_files)} files successfully!")

# %% Cell 5: Prepare Watermark Payloads
print("Preparing watermark payloads...")

# Create metadata for each file
metadata_list = []
for i in range(len(preprocessed_files)):
    metadata = {
        'patient_id': np.random.randint(0, 256),
        'doctor_id': np.random.randint(0, 256),
        'date_stamp': np.random.randint(0, 65536),
        'diagnosis_code': np.random.randint(0, 256),
        'hospital_id': np.random.randint(0, 256)
    }
    metadata_list.append(metadata)

print(f"Created metadata for {len(metadata_list)} files")
print(f"Sample metadata: {metadata_list[0]}")

# %% [markdown]
# ## PHASE 2: Module 1 - TEO Enhanced Frame Classification

# %% Cell 6: Test Frame Classification on Sample Audio
print("Testing TEO-enhanced frame classification...")

# Load sample audio
sample_audio, _ = load_audio(preprocessed_files[0])

# Classify frames
frames, voiced_indices, unvoiced_indices = classify_frames_teo(sample_audio)

print(f"Total frames: {len(frames)}")
print(f"Voiced frames: {len(voiced_indices)} ({len(voiced_indices)/len(frames)*100:.1f}%)")
print(f"Unvoiced frames: {len(unvoiced_indices)} ({len(unvoiced_indices)/len(frames)*100:.1f}%)")

# Visualize
plt.figure(figsize=(12, 4))
plt.plot(sample_audio, alpha=0.7, label='Audio Signal')
plt.title('Sample Audio with Frame Classification')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'frame_classification.png'))
plt.close()
print("Frame classification visualization saved!")

# %% [markdown]
# ## PHASE 3: Module 2 - Energy-Aware DWT-SVD Embedding

# %% Cell 7: Embed Watermark in Sample Audio
print("Embedding watermark using DWT-SVD...")

# Prepare watermark
metadata = metadata_list[0]
watermark_bits = metadata_to_bits(
    metadata['patient_id'],
    metadata['doctor_id'],
    metadata['date_stamp'],
    metadata['diagnosis_code'],
    metadata['hospital_id']
)

print(f"Watermark bits: {watermark_bits[:16]}... (showing first 16 bits)")

# Embed in voiced frames
secret_key = 0.5
watermarked_frames = embed_watermark_voiced_frames(
    frames, voiced_indices, watermark_bits, secret_key
)

# Reconstruct audio
frame_length = int(FRAME_LENGTH_MS * SAMPLE_RATE / 1000)
hop_length = int(frame_length * (1 - FRAME_OVERLAP))
watermarked_audio_intermediate = np.zeros_like(sample_audio)

for i, frame in enumerate(watermarked_frames):
    start = i * hop_length
    end = start + len(frame)
    if end <= len(watermarked_audio_intermediate):
        watermarked_audio_intermediate[start:end] += frame

# Normalize
watermarked_audio_intermediate = watermarked_audio_intermediate / np.max(np.abs(watermarked_audio_intermediate))

print("DWT-SVD embedding complete!")
snr_dwt = calculate_snr(sample_audio, watermarked_audio_intermediate)
print(f"SNR after DWT-SVD embedding: {snr_dwt:.2f} dB")

# %% [markdown]
# ## PHASE 4: Module 3 - DCT Fragile Watermarking

# %% Cell 8: Embed Fragile Watermark
print("Embedding fragile watermark using DCT...")

# Embed in unvoiced frames
watermarked_frames_final = embed_fragile_watermark_unvoiced(
    watermarked_frames, unvoiced_indices, watermarked_audio_intermediate
)

# Reconstruct final audio
watermarked_audio_final = np.zeros_like(sample_audio)
for i, frame in enumerate(watermarked_frames_final):
    start = i * hop_length
    end = start + len(frame)
    if end <= len(watermarked_audio_final):
        watermarked_audio_final[start:end] += frame

watermarked_audio_final = watermarked_audio_final / np.max(np.abs(watermarked_audio_final))

print("DCT fragile embedding complete!")
snr_final = calculate_snr(sample_audio, watermarked_audio_final)
print(f"Final SNR: {snr_final:.2f} dB")

# Save watermarked audio
save_audio(watermarked_audio_final, os.path.join(WATERMARKED_DIR, 'sample_watermarked.wav'))
print("Watermarked audio saved!")

# %% [markdown]
# ## PHASE 5: Module 4 - Multi-Scale Lyapunov Exponent Verification

# %% Cell 9: Build MSLE Signature
print("Building Multi-Scale Lyapunov Exponent signature...")

msle_signature = build_msle_signature(watermarked_audio_final)
print(f"MSLE signature: {msle_signature}")
print(f"Number of blocks: {len(msle_signature)}")

# Visualize MSLE signature
plt.figure(figsize=(10, 4))
plt.bar(range(len(msle_signature)), msle_signature)
plt.title('Multi-Scale Lyapunov Exponent Signature')
plt.xlabel('Block Index')
plt.ylabel('MSLE Value')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'msle_signature.png'))
plt.close()
print("MSLE signature visualization saved!")

# %% [markdown]
# ## PHASE 6: Watermark Extraction

# %% Cell 10: Extract Watermark from Clean Audio
print("Extracting watermark from clean watermarked audio...")

extracted_robust, extracted_fragile = extract_all_watermarks(watermarked_audio_final, secret_key)

print(f"Original watermark: {watermark_bits[:16]}...")
print(f"Extracted watermark: {extracted_robust[:16]}...")

ber_clean = calculate_ber(watermark_bits, extracted_robust)
nc_clean = calculate_nc(watermark_bits, extracted_robust)

print(f"BER (clean): {ber_clean:.4f}")
print(f"NC (clean): {nc_clean:.4f}")

# Verify metadata
extracted_metadata = bits_to_metadata(extracted_robust)
print(f"Original metadata: {metadata}")
print(f"Extracted metadata: Patient ID={extracted_metadata[0]}, Doctor ID={extracted_metadata[1]}")

# %% [markdown]
# ## PHASE 7: Attack Simulation

# %% Cell 11: Apply Attacks and Measure Robustness
print("Applying attacks and measuring robustness...")

attack_types = [
    'awgn_20', 'awgn_10', 
    'pitch_shift', 'time_stretch', 'lowpass', 
    'resampling', 'amplitude'
]

results = []

for attack_type in tqdm(attack_types, desc="Testing attacks"):
    # Apply attack
    attacked_audio = apply_attack(watermarked_audio_final, attack_type)
    
    # Extract watermark
    try:
        extracted_bits, _ = extract_all_watermarks(attacked_audio, secret_key)
        
        # Calculate metrics
        ber = calculate_ber(watermark_bits, extracted_bits)
        nc = calculate_nc(watermark_bits, extracted_bits)
        snr = calculate_snr(watermarked_audio_final, attacked_audio)
        
        results.append({
            'Attack': attack_type,
            'BER': ber,
            'NC': nc,
            'SNR': snr
        })
        
        print(f"{attack_type}: BER={ber:.4f}, NC={nc:.4f}, SNR={snr:.2f}dB")
    except Exception as e:
        print(f"Error with {attack_type}: {e}")
        results.append({
            'Attack': attack_type,
            'BER': 1.0,
            'NC': 0.0,
            'SNR': 0.0
        })

# Create results dataframe
results_df = pd.DataFrame(results)
print("\nAttack Results Summary:")
print(results_df)

# Save results
results_df.to_csv(os.path.join(RESULTS_DIR, 'attack_results.csv'), index=False)
print("Attack results saved!")

# %% Cell 12: Visualize Attack Results
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.bar(results_df['Attack'], results_df['BER'])
plt.title('Bit Error Rate (BER) Under Different Attacks')
plt.xlabel('Attack Type')
plt.ylabel('BER')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.05, color='r', linestyle='--', label='Target Threshold')
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.bar(results_df['Attack'], results_df['NC'])
plt.title('Normalized Correlation (NC) Under Different Attacks')
plt.xlabel('Attack Type')
plt.ylabel('NC')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.90, color='r', linestyle='--', label='Target Threshold')
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, 'attack_results_visualization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Attack results visualization saved!")

# %% Cell 13: Test Tampering Detection with MSLE
print("Testing tampering detection using MSLE...")

# Apply tampering attack
tampered_audio = tampering_attack(watermarked_audio_final, start_sec=2.0, duration_sec=1.0)

# Verify integrity with adaptive threshold
is_authentic, tampered_blocks = verify_msle_integrity(msle_signature, tampered_audio, threshold=None)

print(f"Is authentic: {is_authentic}")
print(f"Tampered blocks detected: {tampered_blocks}")

if len(tampered_blocks) > 0:
    print(f"Tampering detected in time range: {tampered_blocks[0]}-{tampered_blocks[-1]+1} seconds")
else:
    print("No tampering detected (this may indicate threshold needs adjustment)")

# Visualize tampering detection
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(watermarked_audio_final, alpha=0.7, label='Original Watermarked')
plt.title('Original Watermarked Audio')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(tampered_audio, alpha=0.7, label='Tampered Audio', color='red')
for block_idx in tampered_blocks:
    start = block_idx * SAMPLE_RATE
    end = (block_idx + 1) * SAMPLE_RATE
    plt.axvspan(start, end, alpha=0.3, color='yellow', label='Detected Tampering' if block_idx == tampered_blocks[0] else '')
plt.title('Tampered Audio with Detection')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'tampering_detection.png'))
plt.close()
print("Tampering detection visualization saved!")

# %% [markdown]
# ## PHASE 8: Performance Evaluation

# %% Cell 14: Calculate Imperceptibility Metrics
print("Calculating imperceptibility metrics...")

# SNR
snr_value = calculate_snr(sample_audio, watermarked_audio_final)
print(f"SNR: {snr_value:.2f} dB (Target: >35 dB)")

# PESQ
try:
    pesq_score = calculate_pesq_score(sample_audio, watermarked_audio_final)
    print(f"PESQ: {pesq_score:.2f} (Target: >3.5)")
except Exception as e:
    print(f"PESQ calculation error: {e}")
    pesq_score = -1.0

# Create imperceptibility summary
imperceptibility_metrics = {
    'Metric': ['SNR (dB)', 'PESQ', 'Target SNR', 'Target PESQ'],
    'Value': [snr_value, pesq_score, '>35', '>3.5']
}

imperceptibility_df = pd.DataFrame(imperceptibility_metrics)
print("\nImperceptibility Metrics:")
print(imperceptibility_df)

imperceptibility_df.to_csv(os.path.join(RESULTS_DIR, 'imperceptibility_metrics.csv'), index=False)

# %% Cell 15: Calculate Integrity Metrics
print("Calculating integrity verification metrics...")

# Test on more files for better statistics (10 instead of 5)
num_test_files = min(10, len(preprocessed_files))
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

print(f"Testing {num_test_files} files for integrity verification...")

for i in range(num_test_files):
    print(f"  Testing file {i+1}/{num_test_files}...", end=' ')
    try:
        # Load and watermark audio
        audio, _ = load_audio(preprocessed_files[i])
        frames, voiced_idx, unvoiced_idx = classify_frames_teo(audio)
        
        # Quick watermarking
        wm_frames = embed_watermark_voiced_frames(frames, voiced_idx, watermark_bits, secret_key)
        wm_audio = np.zeros_like(audio)
        for j, frame in enumerate(wm_frames):
            start = j * hop_length
            end = start + len(frame)
            if end <= len(wm_audio):
                wm_audio[start:end] += frame
        wm_audio = wm_audio / (np.max(np.abs(wm_audio)) + 1e-8)
        
        # Build signature
        signature = build_msle_signature(wm_audio)
        
        # Test authentic audio (use adaptive threshold)
        is_auth, _ = verify_msle_integrity(signature, wm_audio, threshold=None)
        if is_auth:
            true_negatives += 1
        else:
            false_positives += 1
        
        # Test tampered audio (use adaptive threshold)
        tampered = tampering_attack(wm_audio, 1.0, 0.5)
        is_auth, detected = verify_msle_integrity(signature, tampered, threshold=None)
        if not is_auth and len(detected) > 0:
            true_positives += 1
            print("PASS")
        else:
            false_negatives += 1
            print("FAIL")
            
    except Exception as e:
        print(f"ERROR: {e}")

# Calculate rates
detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

print(f"\nTamper Detection Rate: {detection_rate*100:.1f}% (Target: >95%)")
print(f"False Positive Rate: {false_positive_rate*100:.1f}%")
print(f"True Positives: {true_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

integrity_metrics = {
    'Metric': ['Detection Rate (%)', 'False Positive Rate (%)', 'True Positives', 'True Negatives'],
    'Value': [detection_rate*100, false_positive_rate*100, true_positives, true_negatives]
}

integrity_df = pd.DataFrame(integrity_metrics)
integrity_df.to_csv(os.path.join(RESULTS_DIR, 'integrity_metrics.csv'), index=False)
print("Integrity metrics saved!")

# %% [markdown]
# ## PHASE 9: Comparison with Baseline Methods

# %% Cell 16: Implement Baseline Methods
print("Implementing baseline watermarking methods...")

def baseline_phase_coding(audio, watermark_bits):
    """Baseline 1: Standard Phase Coding"""
    fft = np.fft.fft(audio)
    phase = np.angle(fft)
    magnitude = np.abs(fft)
    
    # Embed in phase
    for i, bit in enumerate(watermark_bits):
        if i < len(phase) // 2:
            if bit == 1:
                phase[i] = np.pi / 4
            else:
                phase[i] = -np.pi / 4
    
    # Reconstruct
    watermarked_fft = magnitude * np.exp(1j * phase)
    watermarked = np.fft.ifft(watermarked_fft).real
    return watermarked

def baseline_dwt_svd_fixed(frames, voiced_indices, watermark_bits):
    """Baseline 2: DWT-SVD with Fixed Alpha"""
    watermarked_frames = frames.copy()
    alpha_fixed = 0.01
    
    for i, frame_idx in enumerate(voiced_indices):
        if i >= len(watermark_bits):
            break
        
        frame = frames[frame_idx]
        coeffs = pywt.wavedec(frame, 'db4', level=2)
        ll = coeffs[0]
        
        # Simple SVD embedding with fixed alpha
        if len(ll) >= 4:
            ll_2d = ll[:4].reshape(2, 2)
            U, S, Vt = np.linalg.svd(ll_2d)
            if watermark_bits[i] == 1:
                S[0] += alpha_fixed
            else:
                S[0] -= alpha_fixed
            modified_ll = np.dot(U, np.dot(np.diag(S), Vt)).flatten()
            coeffs[0][:4] = modified_ll
        
        watermarked_frames[frame_idx] = pywt.waverec(coeffs, 'db4')[:len(frame)]
    
    return watermarked_frames

def baseline_lsb(audio, watermark_bits):
    """Baseline 3: LSB Audio Watermarking"""
    # Convert to 16-bit integers
    audio_int = (audio * 32767).astype(np.int16)
    
    # Embed in LSB
    for i, bit in enumerate(watermark_bits):
        if i < len(audio_int):
            if bit == 1:
                audio_int[i] = audio_int[i] | 1
            else:
                audio_int[i] = audio_int[i] & ~1
    
    # Convert back
    watermarked = audio_int.astype(np.float32) / 32767
    return watermarked

def baseline_echo_hiding(audio, watermark_bits, delay=100, alpha=0.5):
    """Baseline 4: Echo Hiding"""
    watermarked = audio.copy()
    
    for i, bit in enumerate(watermark_bits):
        if i * 1000 + delay < len(audio):
            if bit == 1:
                watermarked[i*1000+delay:i*1000+delay+100] += alpha * audio[i*1000:i*1000+100]
    
    return watermarked / np.max(np.abs(watermarked))

print("Baseline methods implemented!")

# %% Cell 17: Compare All Methods
print("Comparing proposed method with baselines...")

comparison_results = []

# Test on sample audio
test_audio = sample_audio.copy()

# Helper function to calculate BER
def calculate_ber_baseline(original_bits, extracted_bits):
    """Calculate BER between original and extracted bits"""
    min_len = min(len(original_bits), len(extracted_bits))
    if min_len == 0:
        return 1.0
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
    return errors / min_len

# Proposed method (already computed)
comparison_results.append({
    'Method': 'Proposed (Energy-Aware DWT-SVD + MSLE)',
    'SNR': snr_final,
    'BER_Clean': ber_clean,
    'NC_Clean': nc_clean
})

# Baseline 1: Phase Coding
try:
    wm_phase = baseline_phase_coding(test_audio, watermark_bits)
    snr_phase = calculate_snr(test_audio, wm_phase)
    
    # Extract from phase coding
    fft_wm = np.fft.fft(wm_phase)
    phase_wm = np.angle(fft_wm)
    extracted_phase = []
    for i in range(len(watermark_bits)):
        if i < len(phase_wm) // 2:
            if phase_wm[i] > 0:
                extracted_phase.append(1)
            else:
                extracted_phase.append(0)
    extracted_phase = np.array(extracted_phase[:len(watermark_bits)])
    ber_phase = calculate_ber_baseline(watermark_bits, extracted_phase)
    nc_phase = 1 - 2 * ber_phase
    
    comparison_results.append({
        'Method': 'Baseline 1: Phase Coding',
        'SNR': snr_phase,
        'BER_Clean': ber_phase,
        'NC_Clean': nc_phase
    })
except Exception as e:
    print(f"Phase coding error: {e}")

# Baseline 2: Fixed Alpha DWT-SVD
try:
    frames_test, voiced_test, _ = classify_frames_teo(test_audio)
    wm_frames_fixed = baseline_dwt_svd_fixed(frames_test, voiced_test, watermark_bits)
    wm_fixed = np.zeros_like(test_audio)
    for i, frame in enumerate(wm_frames_fixed):
        start = i * hop_length
        end = start + len(frame)
        if end <= len(wm_fixed):
            wm_fixed[start:end] += frame
    wm_fixed = wm_fixed / (np.max(np.abs(wm_fixed)) + 1e-8)
    snr_fixed = calculate_snr(test_audio, wm_fixed)
    
    # Extract from fixed alpha DWT-SVD
    frames_wm, voiced_wm, _ = classify_frames_teo(wm_fixed)
    extracted_fixed = []
    for i, frame_idx in enumerate(voiced_wm):
        if i >= len(watermark_bits):
            break
        frame = frames_wm[frame_idx]
        coeffs = pywt.wavedec(frame, 'db4', level=2)
        ll = coeffs[0]
        if len(ll) >= 4:
            ll_2d = ll[:4].reshape(2, 2)
            _, S, _ = np.linalg.svd(ll_2d)
            # Simple threshold extraction
            if i > 0 and len(extracted_fixed) > 0:
                # Compare with previous
                prev_coeffs = pywt.wavedec(frames_wm[voiced_wm[i-1]], 'db4', level=2)
                prev_ll = prev_coeffs[0][:4].reshape(2, 2)
                _, prev_S, _ = np.linalg.svd(prev_ll)
                if S[0] > prev_S[0]:
                    extracted_fixed.append(1)
                else:
                    extracted_fixed.append(0)
            else:
                extracted_fixed.append(0)
    extracted_fixed = np.array(extracted_fixed[:len(watermark_bits)])
    if len(extracted_fixed) < len(watermark_bits):
        extracted_fixed = np.pad(extracted_fixed, (0, len(watermark_bits) - len(extracted_fixed)))
    ber_fixed = calculate_ber_baseline(watermark_bits, extracted_fixed)
    nc_fixed = 1 - 2 * ber_fixed
    
    comparison_results.append({
        'Method': 'Baseline 2: DWT-SVD Fixed Alpha',
        'SNR': snr_fixed,
        'BER_Clean': ber_fixed,
        'NC_Clean': nc_fixed
    })
except Exception as e:
    print(f"Fixed alpha error: {e}")

# Baseline 3: LSB
try:
    wm_lsb = baseline_lsb(test_audio, watermark_bits)
    snr_lsb = calculate_snr(test_audio, wm_lsb)
    
    # Extract from LSB
    audio_int_wm = (wm_lsb * 32767).astype(np.int16)
    extracted_lsb = []
    for i in range(len(watermark_bits)):
        if i < len(audio_int_wm):
            extracted_lsb.append(audio_int_wm[i] & 1)
    extracted_lsb = np.array(extracted_lsb[:len(watermark_bits)])
    ber_lsb = calculate_ber_baseline(watermark_bits, extracted_lsb)
    nc_lsb = 1 - 2 * ber_lsb
    
    comparison_results.append({
        'Method': 'Baseline 3: LSB',
        'SNR': snr_lsb,
        'BER_Clean': ber_lsb,
        'NC_Clean': nc_lsb
    })
except Exception as e:
    print(f"LSB error: {e}")

# Baseline 4: Echo Hiding
try:
    wm_echo = baseline_echo_hiding(test_audio, watermark_bits)
    snr_echo = calculate_snr(test_audio, wm_echo)
    
    # Extract from echo hiding (simplified detection)
    extracted_echo = []
    for i in range(len(watermark_bits)):
        if i * 1000 + 100 < len(wm_echo):
            # Detect echo presence
            segment = wm_echo[i*1000+100:i*1000+200]
            energy = np.sum(segment ** 2)
            if energy > 0.01:
                extracted_echo.append(1)
            else:
                extracted_echo.append(0)
    extracted_echo = np.array(extracted_echo[:len(watermark_bits)])
    if len(extracted_echo) < len(watermark_bits):
        extracted_echo = np.pad(extracted_echo, (0, len(watermark_bits) - len(extracted_echo)))
    ber_echo = calculate_ber_baseline(watermark_bits, extracted_echo)
    nc_echo = 1 - 2 * ber_echo
    
    comparison_results.append({
        'Method': 'Baseline 4: Echo Hiding',
        'SNR': snr_echo,
        'BER_Clean': ber_echo,
        'NC_Clean': nc_echo
    })
except Exception as e:
    print(f"Echo hiding error: {e}")

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_results)
print("\nMethod Comparison:")
print(comparison_df)

comparison_df.to_csv(os.path.join(RESULTS_DIR, 'method_comparison.csv'), index=False)
print("Comparison results saved!")

# %% Cell 18: Visualize Method Comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
methods = [r.split(':')[0] if ':' in r else r for r in comparison_df['Method']]
plt.barh(methods, comparison_df['SNR'])
plt.xlabel('SNR (dB)')
plt.title('SNR Comparison Across Methods')
plt.axvline(x=35, color='r', linestyle='--', label='Target')
plt.legend()

plt.subplot(1, 2, 2)
plt.barh(methods, comparison_df['NC_Clean'])
plt.xlabel('Normalized Correlation')
plt.title('NC Comparison (Clean Audio)')
plt.axvline(x=0.90, color='r', linestyle='--', label='Target')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'method_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Method comparison visualization saved!")

# %% Cell 19: Generate Final Summary Report
print("Generating final summary report...")

summary_report = f"""
{'='*80}
MEDICAL SPEECH WATERMARKING SYSTEM - FINAL REPORT
{'='*80}

DATASET INFORMATION:
- Total files processed: {len(preprocessed_files)}
- Sample rate: {SAMPLE_RATE} Hz
- Audio duration: {AUDIO_DURATION} seconds
- Watermark payload: {TOTAL_WATERMARK_BITS} bits

IMPERCEPTIBILITY METRICS:
- SNR: {snr_value:.2f} dB (Target: >35 dB) {'PASS' if snr_value > 35 else 'FAIL'}
- PESQ: {pesq_score:.2f} (Target: >3.5) {'PASS' if pesq_score > 3.5 else 'FAIL'}

ROBUSTNESS METRICS (Average):
- BER (Clean): {ber_clean:.4f} (Target: <0.05) {'PASS' if ber_clean < 0.05 else 'FAIL'}
- NC (Clean): {nc_clean:.4f} (Target: >0.90) {'PASS' if nc_clean > 0.90 else 'FAIL'}

INTEGRITY VERIFICATION:
- Tamper Detection Rate: {detection_rate*100:.1f}% (Target: >95%) {'PASS' if detection_rate > 0.95 else 'FAIL'}
- False Positive Rate: {false_positive_rate*100:.1f}%

ATTACK ROBUSTNESS SUMMARY:
"""

for _, row in results_df.iterrows():
    summary_report += f"- {row['Attack']}: BER={row['BER']:.4f}, NC={row['NC']:.4f}\n"

summary_report += f"""
NOVEL CONTRIBUTIONS:
1. TEO-Enhanced Frame Classification for dysarthric speech
2. Energy-Aware Adaptive Alpha in DWT-SVD embedding
3. Pair-Relationship DCT fragile watermarking
4. Multi-Scale Lyapunov Exponent integrity verification

OUTPUT FILES GENERATED:
- Preprocessed audio: {PREPROCESSED_DIR}/
- Watermarked audio: {WATERMARKED_DIR}/
- Results and metrics: {RESULTS_DIR}/
- Visualizations: {RESULTS_DIR}/*.png
- CSV reports: {RESULTS_DIR}/*.csv

{'='*80}
"""

print(summary_report)

# Save report
with open(os.path.join(RESULTS_DIR, 'final_report.txt'), 'w', encoding='utf-8') as f:
    f.write(summary_report)

print("Final report saved to results/final_report.txt")

# %% Cell 20: Create Summary Visualization Dashboard
print("Creating summary dashboard...")

fig = plt.figure(figsize=(16, 10))

# 1. Frame Classification
ax1 = plt.subplot(3, 3, 1)
labels = ['Voiced', 'Unvoiced']
sizes = [len(voiced_indices), len(unvoiced_indices)]
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Frame Classification Distribution')

# 2. SNR Comparison
ax2 = plt.subplot(3, 3, 2)
methods_short = ['Proposed', 'Phase', 'Fixed-α', 'LSB', 'Echo']
snr_values = comparison_df['SNR'].tolist()
ax2.bar(methods_short[:len(snr_values)], snr_values)
ax2.axhline(y=35, color='r', linestyle='--', label='Target')
ax2.set_ylabel('SNR (dB)')
ax2.set_title('SNR Comparison')
ax2.legend()
plt.xticks(rotation=45)

# 3. BER Under Attacks
ax3 = plt.subplot(3, 3, 3)
attack_names = [a.replace('_', '\n') for a in results_df['Attack']]
ax3.bar(range(len(results_df)), results_df['BER'])
ax3.axhline(y=0.05, color='r', linestyle='--', label='Target')
ax3.set_ylabel('BER')
ax3.set_title('BER Under Different Attacks')
ax3.set_xticks(range(len(results_df)))
ax3.set_xticklabels(attack_names, rotation=45, ha='right', fontsize=8)
ax3.legend()

# 4. NC Under Attacks
ax4 = plt.subplot(3, 3, 4)
ax4.bar(range(len(results_df)), results_df['NC'])
ax4.axhline(y=0.90, color='r', linestyle='--', label='Target')
ax4.set_ylabel('NC')
ax4.set_title('NC Under Different Attacks')
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(attack_names, rotation=45, ha='right', fontsize=8)
ax4.legend()

# 5. MSLE Signature
ax5 = plt.subplot(3, 3, 5)
ax5.plot(msle_signature, marker='o')
ax5.set_xlabel('Block Index')
ax5.set_ylabel('MSLE Value')
ax5.set_title('MSLE Signature Profile')
ax5.grid(True, alpha=0.3)

# 6. Integrity Metrics
ax6 = plt.subplot(3, 3, 6)
integrity_labels = ['Detection\nRate', 'False\nPositive']
integrity_values = [detection_rate*100, false_positive_rate*100]
colors = ['green', 'red']
ax6.bar(integrity_labels, integrity_values, color=colors, alpha=0.7)
ax6.set_ylabel('Percentage (%)')
ax6.set_title('Integrity Verification Metrics')
ax6.axhline(y=95, color='b', linestyle='--', label='Target')
ax6.legend()

# 7. Watermark Bit Accuracy
ax7 = plt.subplot(3, 3, 7)
correct_bits = np.sum(watermark_bits == extracted_robust)
incorrect_bits = len(watermark_bits) - correct_bits
ax7.pie([correct_bits, incorrect_bits], labels=['Correct', 'Incorrect'], 
        autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
ax7.set_title('Watermark Extraction Accuracy')

# 8. Audio Waveform Comparison
ax8 = plt.subplot(3, 3, 8)
time_axis = np.linspace(0, AUDIO_DURATION, len(sample_audio))
ax8.plot(time_axis, sample_audio, alpha=0.5, label='Original')
ax8.plot(time_axis, watermarked_audio_final, alpha=0.5, label='Watermarked')
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Amplitude')
ax8.set_title('Original vs Watermarked Audio')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Performance Summary Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
PERFORMANCE SUMMARY

Imperceptibility:
  SNR: {snr_value:.2f} dB
  PESQ: {pesq_score:.2f}

Robustness:
  BER: {ber_clean:.4f}
  NC: {nc_clean:.4f}

Integrity:
  Detection: {detection_rate*100:.1f}%
  False Pos: {false_positive_rate*100:.1f}%

Status: {'PASS ✓' if snr_value > 35 and ber_clean < 0.05 and detection_rate > 0.95 else 'REVIEW'}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'summary_dashboard.png'), dpi=200, bbox_inches='tight')
plt.close()

print("Summary dashboard saved!")
print("\n" + "="*80)
print("ALL PHASES COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nCheck the '{RESULTS_DIR}' directory for all outputs and visualizations.")
