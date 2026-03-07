# Medical Speech Watermarking System

Implementation of adaptive dual-layer watermarking for medical speech recordings using DWT-SVD embedding and multi-scale Lyapunov exponent verification.

## Overview

This system embeds patient metadata into dysarthric speech recordings while maintaining audio quality and detecting tampering. Designed for the TORGO dataset of dysarthric speech.

## Features

- TEO-enhanced frame classification for dysarthric speech
- Energy-aware DWT-SVD robust watermarking
- DCT-based fragile watermarking for tamper detection
- Multi-scale Lyapunov exponent integrity verification
- Chaotic encryption for security
- Repetition coding for error correction

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd medical-speech-watermarking

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

Place TORGO dataset WAV files in the `data/` directory, then run:

```bash
python main_notebook.py
```

Results will be saved to `outputs/results/` including:
- Watermarked audio files
- Performance metrics (SNR, PESQ, BER, NC)
- Attack robustness analysis
- Tamper detection results
- Visualizations

## Configuration

Edit `src/config.py` to customize:
- Audio parameters (sample rate, duration, frame length)
- Watermark payload (48 bits: patient ID, doctor ID, date, diagnosis, hospital)
- DWT-SVD parameters (wavelet type, decomposition level, alpha range)
- MSLE parameters (block duration, window sizes)

## Performance

- SNR: 27.7 dB (imperceptible)
- PESQ: 4.45 (excellent quality)
- BER: 0.17 (with 5x repetition coding)
- Tamper Detection: 100%
- Voiced Frame Accuracy: 65.2%

## Project Structure

```
.
├── main_notebook.py          # Main execution pipeline
├── requirements.txt          # Dependencies
├── src/
│   ├── config.py            # Configuration
│   ├── module1_teo_classification.py
│   ├── module2_dwt_svd_embedding.py
│   ├── module3_dct_fragile.py
│   ├── module4_msle_verification.py
│   ├── extraction.py
│   ├── attacks.py
│   └── metrics.py
├── data/                    # TORGO dataset (not tracked)
└── outputs/                 # Results (not tracked)
```

## Dataset

Download the TORGO dataset from: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

Place WAV files in the `data/` directory. The system will automatically select 200 files for processing.

## Git Setup

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR-USERNAME/medical-speech-watermarking.git
git branch -M main
git push -u origin main
```

## License

MIT License - See LICENSE file for details
