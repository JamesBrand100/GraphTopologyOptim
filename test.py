import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version PyTorch was built with:", torch.version.cuda)

#also run this to check if good:
#python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): print(f'GPU Name: {torch.cuda.get_device_name(0)}')"
