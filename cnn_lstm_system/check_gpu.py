import torch

print("="*60)
print("GPU DETECTION")
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPU Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    print(f"\nCurrent Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("\n[OK] GPU is available and ready to use!")
else:
    print("\n[WARNING] No GPU detected. Training will use CPU.")
    print("To use GPU, install CUDA-enabled PyTorch:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("="*60)
