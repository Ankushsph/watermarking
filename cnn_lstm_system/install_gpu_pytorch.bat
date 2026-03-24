@echo off
echo ============================================================
echo Installing CUDA-enabled PyTorch for GPU Training
echo ============================================================
echo.
echo Your GPU: NVIDIA GeForce RTX 4050 (6GB VRAM)
echo CUDA Version: 13.0
echo.
echo This will install PyTorch with CUDA 12.1 support
echo (Compatible with your CUDA 13.0)
echo.
pause

echo.
echo Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Verifying GPU detection...
python check_gpu.py

echo.
pause
