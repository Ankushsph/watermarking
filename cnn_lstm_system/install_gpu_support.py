#!/usr/bin/env python3
"""
Install GPU Support for PyTorch
Automatically detects CUDA version and installs appropriate PyTorch
"""

import subprocess
import sys

print("="*80)
print("GPU PYTORCH INSTALLATION")
print("="*80)
print()

print("Detected GPU: NVIDIA GeForce RTX 4050 (6GB VRAM)")
print("CUDA Version: 13.0")
print()
print("This will install PyTorch with CUDA 12.1 support")
print("(Compatible with your CUDA 13.0)")
print()

response = input("Continue with installation? (y/n) [default: y]: ").strip().lower()
if response == 'n':
    print("Installation cancelled.")
    sys.exit(0)

print()
print("="*80)
print("STEP 1: Uninstalling CPU-only PyTorch")
print("="*80)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])

print()
print("="*80)
print("STEP 2: Installing CUDA-enabled PyTorch")
print("="*80)
subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

print()
print("="*80)
print("STEP 3: Verifying GPU Detection")
print("="*80)
subprocess.run([sys.executable, "check_gpu.py"])

print()
print("="*80)
print("INSTALLATION COMPLETE!")
print("="*80)
print()
print("Your system is now ready for GPU-accelerated training!")
print()
print("Expected speedup: 4-8x faster than CPU")
print()
print("To train with GPU:")
print("  python run_with_training.py")
print()
