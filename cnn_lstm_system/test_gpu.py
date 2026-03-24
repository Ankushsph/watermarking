try:
    import torch
    print("PyTorch installed:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Ready!")
    else:
        print("GPU not detected - using CPU")
except Exception as e:
    print("Error:", e)
