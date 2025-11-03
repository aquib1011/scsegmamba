"""
Benchmark inference performance for SCSegamba on a single image.

This script loads a checkpoint, runs inference on a single image, and measures:
- Inference time
- Memory usage (CPU and GPU)
- Power consumption (GPU)
- GPU utilization
- CPU utilization
- Energy per inference

Usage:
    python benchmark_inference.py --checkpoint checkpoint_Deepcrack.pth --image path/to/image.jpg

Optional dependencies for enhanced GPU monitoring:
    pip install nvidia-ml-py3  # For GPU power and utilization monitoring
"""

import os
import argparse
import time
import numpy as np
import torch
import cv2
import psutil
import gc
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Optional

from models import build_model
from main import get_args_parser


class PerformanceMonitor:
    """Monitor system performance metrics during inference."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.is_gpu = device.type == 'cuda'
        self.process = psutil.Process()
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        self.nvml_initialized = False
        if self.is_gpu and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_available = True
                self.nvml_initialized = True
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device.index if device.index is not None else 0)
            except ImportError:
                print("Warning: pynvml not available. GPU power/util monitoring will be limited.")
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        memory_usage = {
            'cpu_memory_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'cpu_memory_percent': self.process.memory_percent()
        }
        
        if self.is_gpu and torch.cuda.is_available():
            memory_usage.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated(self.device) / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved(self.device) / 1024 / 1024,
            })
            if hasattr(torch.cuda, 'max_memory_allocated'):
                memory_usage['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        
        return memory_usage
    
    def get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        if not self.gpu_available or not self.nvml_initialized:
            return None
        
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu
        except Exception as e:
            print(f"Warning: Could not get GPU utilization: {e}")
            return None
    
    def get_gpu_power(self) -> Optional[float]:
        """Get GPU power consumption in watts."""
        if not self.gpu_available or not self.nvml_initialized:
            return None
        
        try:
            import pynvml
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            return power / 1000.0  # Convert mW to W
        except Exception as e:
            print(f"Warning: Could not get GPU power: {e}")
            return None
    
    def get_cpu_utilization(self) -> float:
        """Get CPU utilization percentage."""
        return self.process.cpu_percent(interval=0.1)
    
    def reset_memory_stats(self):
        """Reset memory statistics for accurate measurement."""
        if self.is_gpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()
        if self.is_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()


def preprocess_image(image_path: str, load_width: int = 512, load_height: int = 512) -> torch.Tensor:
    """Preprocess a single image for inference."""
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    if load_width > 0 and load_height > 0:
        img = cv2.resize(img, (load_width, load_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply same transforms as dataset
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = img_transforms(Image.fromarray(img.copy()))
    return img_tensor.unsqueeze(0)  # Add batch dimension


def benchmark_inference(
    checkpoint_path: str,
    image_path: str,
    device: str = 'cuda',
    load_width: int = 512,
    load_height: int = 512,
    num_warmup: int = 3,
    num_iterations: int = 10
) -> Dict:
    """Benchmark inference performance on a single image."""
    
    print("=" * 80)
    print("SCSegamba Inference Benchmark")
    print("=" * 80)
    
    # Setup device
    device_obj = torch.device(device)
    print(f"Device: {device_obj}")
    
    # Initialize monitor
    monitor = PerformanceMonitor(device_obj)
    
    # Load and preprocess image
    print(f"\nLoading image: {image_path}")
    print(f"Image size: {load_width}x{load_height}")
    image_tensor = preprocess_image(image_path, load_width, load_height)
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Build model
    print("\nBuilding model...")
    args = argparse.Namespace(
        device=device,
        BCELoss_ratio=0.83,
        DiceLoss_ratio=0.17,
        Norm_Type='GN',
        load_width=load_width,
        load_height=load_height
    )
    model, _ = build_model(args)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"State dict load info: {missing}")
    
    model.to(device_obj)
    model.eval()
    print("Model loaded successfully!")
    
    # Move image to device
    image_tensor = image_tensor.to(device_obj)
    
    # Warmup
    print(f"\nWarming up with {num_warmup} iterations...")
    with torch.no_grad():
        for i in range(num_warmup):
            _ = model(image_tensor)
            if i < num_warmup - 1 and device_obj.type == 'cuda':
                torch.cuda.empty_cache()
    print("Warmup completed!")
    
    # Reset memory stats after warmup
    monitor.reset_memory_stats()
    
    # Benchmark measurements
    print(f"\nRunning benchmark with {num_iterations} iterations...")
    inference_times = []
    memory_usages = []
    gpu_utils = []
    cpu_utils = []
    gpu_powers = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            # Measure before inference
            memory_before = monitor.get_memory_usage()
            cpu_util_before = monitor.get_cpu_utilization()
            gpu_util_before = monitor.get_gpu_utilization()
            gpu_power_before = monitor.get_gpu_power()
            
            # Inference
            torch.cuda.synchronize() if device_obj.type == 'cuda' else None
            start_time = time.perf_counter()
            
            output = model(image_tensor)
            
            torch.cuda.synchronize() if device_obj.type == 'cuda' else None
            end_time = time.perf_counter()
            
            # Measure after inference
            memory_after = monitor.get_memory_usage()
            cpu_util_after = monitor.get_cpu_utilization()
            gpu_util_after = monitor.get_gpu_utilization()
            gpu_power_after = monitor.get_gpu_power()
            
            # Record metrics
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            memory_usages.append({
                'before': memory_before,
                'after': memory_after,
                'delta': {k: memory_after.get(k, 0) - memory_before.get(k, 0) 
                         for k in memory_after.keys()}
            })
            
            gpu_utils.append(gpu_util_after if gpu_util_after is not None else 0)
            cpu_utils.append(cpu_util_after)
            gpu_powers.append(gpu_power_after if gpu_power_after is not None else 0)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations...")
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    
    avg_gpu_memory = np.mean([m['after'].get('gpu_memory_allocated_mb', 0) for m in memory_usages])
    peak_gpu_memory = max([m['after'].get('gpu_memory_allocated_mb', 0) for m in memory_usages])
    avg_cpu_memory = np.mean([m['after'].get('cpu_memory_mb', 0) for m in memory_usages])
    
    avg_gpu_util = np.mean([u for u in gpu_utils if u > 0]) if any(u > 0 for u in gpu_utils) else 0
    avg_cpu_util = np.mean(cpu_utils)
    
    avg_gpu_power = np.mean([p for p in gpu_powers if p > 0]) if any(p > 0 for p in gpu_powers) else 0
    
    # Calculate energy per inference
    # Energy (Joules) = Power (Watts) * Time (seconds)
    energy_per_inference = avg_gpu_power * avg_inference_time if avg_gpu_power > 0 else 0
    
    # Compile results
    results = {
        'inference_time': {
            'mean': avg_inference_time,
            'std': std_inference_time,
            'min': min_inference_time,
            'max': max_inference_time,
            'all': inference_times
        },
        'memory_usage': {
            'gpu_memory_mb': avg_gpu_memory,
            'gpu_memory_peak_mb': peak_gpu_memory,
            'cpu_memory_mb': avg_cpu_memory,
            'all': memory_usages
        },
        'gpu_utilization': {
            'mean': avg_gpu_util,
            'all': gpu_utils
        },
        'cpu_utilization': {
            'mean': avg_cpu_util,
            'all': cpu_utils
        },
        'gpu_power': {
            'mean': avg_gpu_power,
            'all': gpu_powers
        },
        'energy_per_inference': {
            'joules': energy_per_inference,
            'millijoules': energy_per_inference * 1000
        }
    }
    
    return results


def print_results(results: Dict):
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Inference Time
    print("\n📊 Inference Performance:")
    print(f"  Mean: {results['inference_time']['mean']*1000:.2f} ms")
    print(f"  Std:  {results['inference_time']['std']*1000:.2f} ms")
    print(f"  Min:  {results['inference_time']['min']*1000:.2f} ms")
    print(f"  Max:  {results['inference_time']['max']*1000:.2f} ms")
    print(f"  FPS:  {1.0/results['inference_time']['mean']:.2f}")
    
    # Memory Usage
    print("\n💾 Memory Usage:")
    if results['memory_usage']['gpu_memory_mb'] > 0:
        print(f"  GPU Memory (allocated): {results['memory_usage']['gpu_memory_mb']:.2f} MB")
        print(f"  GPU Memory (peak):     {results['memory_usage']['gpu_memory_peak_mb']:.2f} MB")
    print(f"  CPU Memory:            {results['memory_usage']['cpu_memory_mb']:.2f} MB")
    
    # GPU Utilization
    if results['gpu_utilization']['mean'] > 0:
        print("\n🎮 GPU Utilization:")
        print(f"  Mean: {results['gpu_utilization']['mean']:.1f}%")
    
    # CPU Utilization
    print("\n🖥️  CPU Utilization:")
    print(f"  Mean: {results['cpu_utilization']['mean']:.1f}%")
    
    # Power Consumption
    if results['gpu_power']['mean'] > 0:
        print("\n⚡ Power Consumption:")
        print(f"  GPU Power (mean): {results['gpu_power']['mean']:.2f} W")
    
    # Energy per Inference
    if results['energy_per_inference']['joules'] > 0:
        print("\n🔋 Energy per Inference:")
        print(f"  Energy: {results['energy_per_inference']['millijoules']:.4f} mJ")
        print(f"  Energy: {results['energy_per_inference']['joules']:.6f} J")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser('SCSegamba Inference Benchmark', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., checkpoint_Deepcrack.pth)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--load_width', type=int, default=512,
                        help='Input image width')
    parser.add_argument('--load_height', type=int, default=512,
                        help='Input image height')
    parser.add_argument('--num_warmup', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of benchmark iterations')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: Save results to JSON file')
    
    args = parser.parse_args()
    
    # Run benchmark
    try:
        results = benchmark_inference(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            device=args.device,
            load_width=args.load_width,
            load_height=args.load_height,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations
        )
        
        # Print results
        print_results(results)
        
        # Save to file if requested
        if args.output:
            import json
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            serializable_results = convert_to_serializable(results)
            with open(args.output, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

