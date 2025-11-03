"""
Run inference using released checkpoints on released datasets and compute metrics.
Now includes comprehensive resource monitoring: energy consumption, inference time, and memory usage.

Usage examples (PowerShell on Windows):
  # Basic inference with resource monitoring (default)
  python run_resources_eval.py --dataset_name DeepCrack \
      --resources_root "C:/Users/Owner/Desktop/SCSegamba/Resources_released/Resources_released" \
      --checkpoint "C:/Users/Owner/Desktop/SCSegamba/Resources_released/Resources_released/Checkpoints/DeepCrack/checkpoint_DeepCrack.pth" \
      --device cpu

  # With custom warmup iterations
  python run_resources_eval.py --dataset_name DeepCrack \
      --resources_root "C:/Users/Owner/Desktop/SCSegamba/Resources_released/Resources_released" \
      --device cuda --warmup_iterations 5

  # Disable resource monitoring for faster execution
  python run_resources_eval.py --dataset_name DeepCrack \
      --resources_root "C:/Users/Owner/Desktop/SCSegamba/Resources_released/Resources_released" \
      --device cpu --no-monitor_resources

If --checkpoint is omitted, it will be inferred from --resources_root and --dataset_name.

Resource monitoring features:
- Inference time measurement (per batch and total)
- CPU and GPU memory usage tracking
- Energy consumption estimation
- Comprehensive reporting with statistics
- Model warmup for accurate measurements
"""

import argparse
import os
import time
import numpy as np
import torch
import cv2
import psutil
import gc
from contextlib import contextmanager
from typing import Dict, List, Tuple

from datasets import create_dataset
from models import build_model
from main import get_args_parser
from util.logger import get_logger
from eval.evaluate import eval as eval_metrics


class ResourceMonitor:
    """Monitor system resources during model inference."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.is_gpu = device.type == 'cuda'
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.initial_cpu_percent = self.process.cpu_percent()
        
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
                'gpu_memory_cached_mb': torch.cuda.memory_cached(self.device) / 1024 / 1024
            })
        
        return memory_usage
    
    def get_energy_usage(self) -> Dict[str, float]:
        """Get energy usage information (approximate)."""
        energy_info = {}
        
        # CPU energy estimation (rough approximation)
        cpu_percent = self.process.cpu_percent()
        energy_info['cpu_energy_estimate'] = cpu_percent * 0.1  # Rough estimate in watts
        
        # GPU energy estimation (if available)
        if self.is_gpu and torch.cuda.is_available():
            try:
                # This is a rough estimation - actual GPU power monitoring requires specific libraries
                gpu_util = torch.cuda.utilization(self.device) if hasattr(torch.cuda, 'utilization') else 50
                energy_info['gpu_energy_estimate'] = gpu_util * 0.2  # Rough estimate in watts
            except:
                energy_info['gpu_energy_estimate'] = 0.0
        
        return energy_info
    
    def reset_memory_stats(self):
        """Reset memory statistics for accurate measurement."""
        if self.is_gpu and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        gc.collect()
        if self.is_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()


@contextmanager
def measure_inference_time():
    """Context manager to measure inference time."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    return end_time - start_time


def warmup_model(model: torch.nn.Module, device: torch.device, data_loader, num_warmup: int = 5):
    """Warm up the model with a few inference runs."""
    print(f"Warming up model with {num_warmup} iterations...")
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_warmup:
                break
                
            x = data["image"]
            if str(device) != 'cpu':
                x = x.cuda()
            
            _ = model(x)
            
            if i < num_warmup - 1:  # Don't clear cache on last iteration
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    print("Model warmup completed!")


def generate_resource_report(metrics: Dict, output_dir: str, device: torch.device) -> Dict[str, float]:
    """Generate comprehensive resource usage report."""
    inference_times = metrics['inference_times']
    memory_usage = metrics['memory_usage']
    energy_usage = metrics['energy_usage']
    total_inferences = metrics['total_inferences']
    
    # Calculate timing statistics
    timing_stats = {
        'total_inference_time': sum(inference_times),
        'avg_inference_time': np.mean(inference_times),
        'min_inference_time': np.min(inference_times),
        'max_inference_time': np.max(inference_times),
        'std_inference_time': np.std(inference_times),
        'inferences_per_second': total_inferences / sum(inference_times) if sum(inference_times) > 0 else 0
    }
    
    # Calculate memory statistics
    memory_stats = {}
    if memory_usage:
        # Get all memory keys
        all_memory_keys = set()
        for mem_dict in memory_usage:
            all_memory_keys.update(mem_dict.keys())
        
        for key in all_memory_keys:
            values = [mem_dict.get(key, 0) for mem_dict in memory_usage if key in mem_dict]
            if values:
                memory_stats[f'{key}_avg'] = np.mean(values)
                memory_stats[f'{key}_max'] = np.max(values)
                memory_stats[f'{key}_min'] = np.min(values)
                memory_stats[f'{key}_std'] = np.std(values)
    
    # Calculate energy statistics
    energy_stats = {}
    if energy_usage:
        # Get all energy keys
        all_energy_keys = set()
        for energy_dict in energy_usage:
            all_energy_keys.update(energy_dict.keys())
        
        for key in all_energy_keys:
            values = [energy_dict.get(key, 0) for energy_dict in energy_usage if key in energy_dict]
            if values:
                energy_stats[f'{key}_avg'] = np.mean(values)
                energy_stats[f'{key}_max'] = np.max(values)
                energy_stats[f'{key}_min'] = np.min(values)
                energy_stats[f'{key}_std'] = np.std(values)
    
    # Combine all statistics
    report = {
        **timing_stats,
        **memory_stats,
        **energy_stats,
        'device_type': str(device),
        'total_inferences': total_inferences
    }
    
    # Save detailed report to file
    report_file = os.path.join(output_dir, 'resource_usage_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RESOURCE USAGE REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Device: {device}\n")
        f.write(f"Total Inferences: {total_inferences}\n\n")
        
        f.write("TIMING STATISTICS:\n")
        f.write("-" * 30 + "\n")
        for key, value in timing_stats.items():
            f.write(f"{key}: {value:.6f}\n")
        
        f.write("\nMEMORY STATISTICS (MB):\n")
        f.write("-" * 30 + "\n")
        for key, value in memory_stats.items():
            f.write(f"{key}: {value:.2f}\n")
        
        f.write("\nENERGY STATISTICS (Watts):\n")
        f.write("-" * 30 + "\n")
        for key, value in energy_stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    return report


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(
        'Eval released resources',
        parents=[get_args_parser()],
        conflict_handler='resolve',   # <-- add this
    )
    parent.add_argument(
        '--resources_root',
        type=str,
        default=os.path.join('.', 'Resources_released', 'Resources_released'),
        help='Root of Resources_released/Resources_released'
    )
    parent.add_argument(
        '--dataset_name',
        type=str,
        default='DeepCrack',
        choices=['DeepCrack', 'Crack500', 'CrackMap'],
        help='Which released dataset/checkpoint to use'
    )
    parent.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to the released checkpoint (*.pth). If omitted, inferred from resources_root.'
    )
    parent.add_argument(
        '--output_dir',               # this now safely overrides the parent's
        type=str,
        default=None,
        help='Directory to save predictions and logs (defaults under ./results)'
    )
    parent.add_argument(
        '--warmup_iterations',
        type=int,
        default=3,
        help='Number of warmup iterations for accurate measurements (default: 3)'
    )
    parent.add_argument(
        '--monitor_resources',
        action='store_true',
        default=True,
        help='Enable comprehensive resource monitoring (default: True)'
    )
    return parent


def infer_and_save(model: torch.nn.Module, device: torch.device, data_loader, save_root: str, 
                  monitor: ResourceMonitor = None) -> Dict[str, List[float]]:
    """Run inference with comprehensive resource monitoring."""
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    
    # Initialize resource tracking
    if monitor is None:
        monitor = ResourceMonitor(device)
    
    # Reset memory stats for accurate measurement
    monitor.reset_memory_stats()
    
    # Track metrics for each inference
    inference_times = []
    memory_usage_per_inference = []
    energy_usage_per_inference = []
    
    print("Starting inference with resource monitoring...")
    
    with torch.no_grad():
        model.eval()
        for batch_idx, data in enumerate(data_loader):
            # Measure memory before inference
            memory_before = monitor.get_memory_usage()
            energy_before = monitor.get_energy_usage()
            
            # Prepare data
            x = data["image"]
            target = data["label"]
            if str(device) != 'cpu':
                x = x.cuda()
                target = target.to(dtype=torch.int64).cuda()

            # Measure inference time
            start_time = time.perf_counter()
            out = model(x)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            # Measure memory and energy after inference
            memory_after = monitor.get_memory_usage()
            energy_after = monitor.get_energy_usage()
            
            # Calculate memory delta
            memory_delta = {}
            for key in memory_after:
                if key in memory_before:
                    memory_delta[f"{key}_delta"] = memory_after[key] - memory_before[key]
                else:
                    memory_delta[key] = memory_after[key]
            
            # Calculate energy delta
            energy_delta = {}
            for key in energy_after:
                if key in energy_before:
                    energy_delta[f"{key}_delta"] = energy_after[key] - energy_before[key]
                else:
                    energy_delta[key] = energy_after[key]
            
            memory_usage_per_inference.append(memory_delta)
            energy_usage_per_inference.append(energy_delta)

            # Process outputs
            target_np = target[0, 0, ...].detach().cpu().numpy()
            out_np = out[0, 0, ...].detach().cpu().numpy()

            root_name = os.path.basename(data["A_paths"][0]).rsplit('.', 1)[0]

            # Safe normalization to 0-255
            t_den = np.max(target_np)
            p_den = np.max(out_np)
            if t_den == 0:
                t_den = 1.0
            if p_den == 0:
                p_den = 1.0
            target_vis = (255.0 * (target_np / t_den)).astype(np.uint8)
            pred_vis = (255.0 * (out_np / p_den)).astype(np.uint8)

            cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), target_vis)
            cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), pred_vis)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches, "
                      f"Avg inference time: {np.mean(inference_times):.4f}s, "
                      f"Current GPU memory: {memory_after.get('gpu_memory_allocated_mb', 0):.2f}MB")
    
    # Return comprehensive metrics
    return {
        'inference_times': inference_times,
        'memory_usage': memory_usage_per_inference,
        'energy_usage': energy_usage_per_inference,
        'total_inferences': len(inference_times)
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Configure dataset args expected by existing pipeline
    args.phase = 'test'
    args.dataset_mode = 'crack'
    args.batch_size = 1

    # Dataset path in released resources
    args.dataset_path = os.path.join(args.resources_root, 'Datasets', args.dataset_name)

    # Checkpoint path
    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            args.resources_root, 'Checkpoints', args.dataset_name, f'checkpoint_{args.dataset_name}.pth'
        )

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Output directory
    if args.output_dir is None:
        time_tag = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        args.output_dir = os.path.join('./results', f'{args.dataset_name}_resources_eval', time_tag)
    os.makedirs(args.output_dir, exist_ok=True)

    # Loggers
    log_eval = get_logger(args.output_dir, 'eval')

    # Device and dataloader
    device = torch.device(args.device)
    data_loader = create_dataset(args)

    # Build model and load released weights
    model, _ = build_model(args)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu',weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing = model.load_state_dict(state_dict, strict=False)
    if getattr(missing, 'missing_keys', None) or getattr(missing, 'unexpected_keys', None):
        print(f"State dict load info: {missing}")

    model.to(device)
    print("Load Model Successful!")

    # Initialize resource monitor if monitoring is enabled
    monitor = None
    if args.monitor_resources:
        monitor = ResourceMonitor(device)
        print(f"Initial memory usage: {monitor.get_memory_usage()}")
        
        # Warm up the model for accurate measurements
        warmup_model(model, device, data_loader, num_warmup=args.warmup_iterations)
    
    # Run inference with or without resource monitoring
    save_root = args.output_dir
    if args.monitor_resources:
        print("\n" + "="*60)
        print("STARTING INFERENCE WITH RESOURCE MONITORING")
        print("="*60)
        
        resource_metrics = infer_and_save(model, device, data_loader, save_root, monitor)
        
        # Generate comprehensive resource report
        print("\nGenerating resource usage report...")
        resource_report = generate_resource_report(resource_metrics, save_root, device)
        
        # Print summary to console
        print("\n" + "="*60)
        print("RESOURCE USAGE SUMMARY")
        print("="*60)
        print(f"Device: {device}")
        print(f"Total Inferences: {resource_report['total_inferences']}")
        print(f"Average Inference Time: {resource_report['avg_inference_time']:.4f}s")
        print(f"Inferences per Second: {resource_report['inferences_per_second']:.2f}")
        print(f"Total Inference Time: {resource_report['total_inference_time']:.2f}s")
        
        # Memory summary
        if 'cpu_memory_mb_avg' in resource_report:
            print(f"Average CPU Memory: {resource_report['cpu_memory_mb_avg']:.2f}MB")
        if 'gpu_memory_allocated_mb_avg' in resource_report:
            print(f"Average GPU Memory Allocated: {resource_report['gpu_memory_allocated_mb_avg']:.2f}MB")
        
        # Energy summary
        if 'cpu_energy_estimate_avg' in resource_report:
            print(f"Average CPU Energy: {resource_report['cpu_energy_estimate_avg']:.4f}W")
        if 'gpu_energy_estimate_avg' in resource_report:
            print(f"Average GPU Energy: {resource_report['gpu_energy_estimate_avg']:.4f}W")
        
        print(f"\nDetailed report saved to: {os.path.join(save_root, 'resource_usage_report.txt')}")
    else:
        print("\n" + "="*60)
        print("STARTING INFERENCE (RESOURCE MONITORING DISABLED)")
        print("="*60)
        infer_and_save(model, device, data_loader, save_root)

    # Compute metrics using existing evaluation
    print("\nComputing evaluation metrics...")
    eval_metrics_result = eval_metrics(log_eval, save_root, epoch=0)
    for key, value in eval_metrics_result.items():
        print(f"{key}: {value}")

    print("\nFinished!")


if __name__ == '__main__':
    main()


