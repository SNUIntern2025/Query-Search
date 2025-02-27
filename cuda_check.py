import torch
import os   
import multiprocessing.shared_memory

import multiprocessing.shared_memory

def cleanup_existing_shared_memory(name_list):
    """이미 생성된 공유 메모리를 정리"""
    for name in name_list:
        try:
            shm = multiprocessing.shared_memory.SharedMemory(name=name)
            shm.unlink()
            print(f"Shared memory {name} unlinked.")
        except FileNotFoundError:
            print(f"Shared memory {name} not found, skipping.")
        except Exception as e:
            print(f"Error cleaning up shared memory {name}: {e}")

        
if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())  # True여야 정상
    print("CUDA Device Count:", torch.cuda.device_count())  # 1 이상이어야 정상
    print("Current CUDA Device:", torch.cuda.current_device())  # 0이어야 정상
    print("CUDA initialized:", torch.cuda.is_initialized()) # False가 나와야 정상
    
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))  # GPU 모델 이름 출력
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    x = torch.tensor([1.0], device="cuda")  # GPU에서 텐서 생성
    print("Memory Allocated:", torch.cuda.memory_allocated())  # 0이 아니어야 정상
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated())  # 최대 메모리 사용량
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version in PyTorch:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
    
    
    # 예제: 기존 공유 메모리 정리
    cleanup_existing_shared_memory(["shm_test1", "shm_test2"])
    
    
    # (base) :
        # CUDA Device Count: 5
        # Current CUDA Device: 0
        # CUDA initialized: True
        # Device Name: NVIDIA A100-PCIE-40GB
        # CUDA_VISIBLE_DEVICES: None
        # Memory Allocated: 512
        # Max Memory Allocated: 512
        # PyTorch Version: 2.5.1+cu124
        # CUDA Version in PyTorch: 12.4
        # cuDNN Version: 90100
        
    # (hyeznee) : 
        # CUDA Device Count: 5
        # Current CUDA Device: 0
        # CUDA initialized: True
        # Device Name: NVIDIA A100-PCIE-40GB
        # CUDA_VISIBLE_DEVICES: None
        # Memory Allocated: 512
        # Max Memory Allocated: 512
        # PyTorch Version: 2.5.1+cu124
        # CUDA Version in PyTorch: 12.4
        # cuDNN Version: 90100
    
    
    