import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

    # Test with a simple tensor operation
    x = torch.rand(5, 3)
    print(f"CPU tensor: {x}")
    x = x.cuda()
    print(f"GPU tensor: {x}")
