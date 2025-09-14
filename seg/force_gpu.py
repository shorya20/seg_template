import os
import sys
import gc
import torch
import numpy as np

def detailed_gpu_info():
    """Get detailed GPU memory information in MB"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    # Get memory information in MB for better precision
    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    
    # Force synchronize to get accurate numbers
    torch.cuda.synchronize()
    
    # Use nvidia-smi for a second opinion
    import subprocess
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used_mem, total_mem = map(int, result.strip().split(','))
    except:
        used_mem, total_mem = -1, -1
    
    return (f"PyTorch CUDA: {allocated_mb:.2f} MB allocated, {reserved_mb:.2f} MB reserved\n"
            f"NVIDIA-SMI: {used_mem} MB used / {total_mem} MB total")

def force_cuda_device(device_id=0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training cannot proceed on GPU.")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # Force CUDA to initialize
    torch.cuda.init()
    
    # Get device information
    device_name = torch.cuda.get_device_name(device_id)
    memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
    memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
    
    print("\n=== GPU CONFIGURATION ===")
    print(f"Using GPU: {device_name}")
    print(f"GPU Memory: {memory_total:.2f} GB")
    print(f"GPU Status: {detailed_gpu_info()}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print("=========================\n")
    
    # Run a verification test with a benchmark operation
    run_gpu_verification_test(device_id)
    
    # Configure for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return torch.device(f'cuda:{device_id}')

def run_gpu_verification_test(device_id=0, n=1000):
    """Run a compute-intensive test on GPU to verify it's working properly"""
    print("Running GPU verification test...")
    try:
        # Create large tensors
        test_size = 2000
        a = torch.randn(test_size, test_size, device=f'cuda:{device_id}', dtype=torch.float32)
        b = torch.randn(test_size, test_size, device=f'cuda:{device_id}', dtype=torch.float32)
        # Benchmark matrix multiplication
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        c = torch.matmul(a, b)
        # Force synchronization
        torch.cuda.synchronize()
        end_time.record()
        # Calculate time
        milliseconds = start_time.elapsed_time(end_time)
        del a, b, c
        torch.cuda.empty_cache()
        
        print(f"GPU Test PASSED: Matrix multiplication completed in {milliseconds:.2f} ms")
        print(f"GPU utilization should now be visible in nvidia-smi")
        
        return True
    except Exception as e:
        print(f"\nGPU Test FAILED: {str(e)}")
        print("The GPU is not being utilized correctly.")
        return False

def diagnose_model_placement(model):
    """Check if a model is correctly placed on GPU"""
    if not isinstance(model, torch.nn.Module):
        print("Not a PyTorch model!")
        return False
    
    gpu_found = False
    cpu_found = False

    for name, param in model.named_parameters():
        if param.device.type == 'cuda':
            gpu_found = True
        elif param.device.type == 'cpu':
            cpu_found = True
            print(f"Parameter found on CPU: {name}")
    
    if gpu_found and cpu_found:
        print("Model has parameters on both CPU and GPU!")
        return False
    elif gpu_found:
        print("Model is properly placed on GPU")
        return True
    else:
        print("Model is on CPU! Use model.to('cuda') to move it to GPU")
        return False

def verify_tensor_on_gpu(tensor, tensor_name="Input"):
    """Verify if a tensor is on GPU and print a helpful message"""
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type == 'cuda':
        return True
    else:
        return False

def force_model_to_gpu(model, device_id=0):
    if not torch.cuda.is_available():
        print("CUDA not available! Cannot move model to GPU.")
        return model
    if isinstance(device_id, torch.device):
        device = device_id
    elif isinstance(device_id, str) and device_id.startswith("cuda"):
        device = torch.device(device_id)
    else:
        device = torch.device(f'cuda:{device_id}')

    all_params_on_gpu = all(p.device.type == 'cuda' for p in model.parameters())
    if all_params_on_gpu:
        print("Model already on GPU")
        return model

    # Move model to GPU
    model = model.to(device)
    all_params_on_gpu = all(p.device.type == 'cuda' for p in model.parameters())
    if all_params_on_gpu:
        print(f"Successfully moved model to GPU: {device}")
    else:
        print("Failed to move model to GPU!")
    return model

def monitor_gpu_usage(message=""):
    if message:
        print(f"=== GPU Usage: {message} ===")
    else:
        print("=== GPU Usage ===")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i)/1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i)/1e9:.2f} GB")
    else:
        print("No GPU available")
