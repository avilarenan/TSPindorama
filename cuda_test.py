import torch

if torch.cuda.is_available():
    print("CUDA (GPU) is available in PyTorch.")
else:
    print("CUDA (GPU) is not available in PyTorch.")

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")
else:
    print("No GPUs detected by PyTorch.")

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.rand(2, 3, device=device)
    print(f"Tensor on GPU: {x}")
else:
    print("GPU not available for tensor creation.")