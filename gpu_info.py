import torch

print("GPU 사용 가능 여부:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 개수:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")