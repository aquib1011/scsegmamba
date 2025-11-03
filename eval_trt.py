"""
Evaluate model on Jetson using TensorRT engine, OR run ONNXRuntime, OR fallback to PyTorch.

It uses the existing dataset loader and metric computation.

Examples (Jetson, TensorRT):
  python eval_trt.py --backend trt --engine ./exports/scsegamba_fp16.plan \
    --dataset_path ./Resources_released/Resources_released/Datasets/DeepCrack \
    --device cuda --height 512 --width 512

Examples (ONNXRuntime):
  python eval_trt.py --backend onnx --onnx ./exports/scsegamba.onnx \
    --dataset_path ./Resources_released/Resources_released/Datasets/DeepCrack \
    --height 512 --width 512

Examples (PyTorch checkpoint):
  python eval_trt.py --backend torch --checkpoint ./Resources_released/Resources_released/Checkpoints/DeepCrack/checkpoint_DeepCrack.pth \
    --dataset_path ./Resources_released/Resources_released/Datasets/DeepCrack --device cuda
"""

import os
import argparse
import time
import numpy as np
import torch
import cv2

from datasets import create_dataset
from models import build_model
from main import get_args_parser
from util.logger import get_logger
from eval.evaluate import eval as eval_metrics


def preprocess_tensor(x: torch.Tensor) -> torch.Tensor:
    # Dataset already normalizes to [-1, 1] via transforms.Normalize(0.5,0.5,0.5)
    return x


class TRTInfer:
    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  # triggers CUDA context

        self.trt = trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine.num_io_tensors == 2, "Expect 1 input and 1 output tensor"
        self.input_name = self.engine.get_tensor_name(0) if self.engine.get_tensor_mode(0) == trt.TensorIOMode.INPUT else self.engine.get_tensor_name(1)
        self.output_name = self.engine.get_tensor_name(1) if self.engine.get_tensor_mode(1) == trt.TensorIOMode.OUTPUT else self.engine.get_tensor_name(0)

        # Allocate device buffers later per-shape
        self.bindings = {}

    def infer(self, x: np.ndarray) -> np.ndarray:
        trt = self.trt
        cuda = self.cuda
        # Ensure NCHW float32
        assert x.ndim == 4 and x.dtype == np.float32
        n, c, h, w = x.shape
        self.context.set_input_shape(self.input_name, (n, c, h, w))
        # Allocate
        d_input = cuda.mem_alloc(x.nbytes)
        self.bindings[self.input_name] = int(d_input)
        # Get output shape from context
        out_shape = self.context.get_tensor_shape(self.output_name)
        if -1 in out_shape:
            # execute once to get dynamic shape resolved
            pass
        out_nbytes = n * 1 * h * w * 4  # 1-channel float32
        d_output = cuda.mem_alloc(out_nbytes)
        self.bindings[self.output_name] = int(d_output)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, x, stream)
        self.context.execute_async_v3(stream.handle)
        out_host = np.empty((n, 1, h, w), dtype=np.float32)
        cuda.memcpy_dtoh_async(out_host, d_output, stream)
        stream.synchronize()
        d_input.free()
        d_output.free()
        return out_host


class ONNXInfer:
    def __init__(self, onnx_path: str, device: str = 'cpu'):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.startswith('cuda') else ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def infer(self, x: np.ndarray) -> np.ndarray:
        out = self.sess.run([self.output_name], {self.input_name: x})[0]
        return out.astype(np.float32)


def infer_and_save_tensors(predict_fn, device: torch.device, data_loader, save_root: str):
    if not os.path.isdir(save_root):
        os.makedirs(save_root)
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            x = data["image"].to(device)
            target = data["label"]
            x = preprocess_tensor(x)
            x_np = x.detach().cpu().numpy() if isinstance(predict_fn, (TRTInfer, ONNXInfer)) else None

            if isinstance(predict_fn, TRTInfer) or isinstance(predict_fn, ONNXInfer):
                out_np = predict_fn.infer(x_np)
                out_t = torch.from_numpy(out_np)
            else:
                out_t = predict_fn(x)

            target_np = target[0, 0, ...].detach().cpu().numpy()
            out_np = out_t[0, 0, ...].detach().cpu().numpy()

            root_name = os.path.basename(data["A_paths"]).rsplit('.', 1)[0] if isinstance(data["A_paths"], str) else os.path.basename(data["A_paths"][0]).rsplit('.', 1)[0]

            t_den = np.max(target_np) or 1.0
            p_den = np.max(out_np) or 1.0
            target_vis = (255.0 * (target_np / t_den)).astype(np.uint8)
            pred_vis = (255.0 * (out_np / p_den)).astype(np.uint8)

            cv2.imwrite(os.path.join(save_root, f"{root_name}_lab.png"), target_vis)
            cv2.imwrite(os.path.join(save_root, f"{root_name}_pre.png"), pred_vis)


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser('Eval with TensorRT/ONNX/Torch', parents=[get_args_parser()], conflict_handler='resolve')
    parent.add_argument('--backend', type=str, default='trt', choices=['trt', 'onnx', 'torch'], help='Inference backend')
    parent.add_argument('--engine', type=str, default=None, help='TensorRT engine path (.plan)')
    parent.add_argument('--onnx', type=str, default=None, help='ONNX model path')
    parent.add_argument('--checkpoint', type=str, default=None, help='PyTorch checkpoint path (.pth) for torch backend')
    parent.add_argument('--output_dir', type=str, default=None, help='Directory to save predictions and logs')
    parent.add_argument('--height', type=int, default=512, help='Input height for backends that need it')
    parent.add_argument('--width', type=int, default=512, help='Input width for backends that need it')
    return parent


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.phase = 'test'
    args.dataset_mode = 'crack'
    args.batch_size = 1

    # Output directory
    if args.output_dir is None:
        time_tag = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        args.output_dir = os.path.join('./results', f'{args.dataset_path.split(os.sep)[-1]}_{args.backend}_eval', time_tag)
    os.makedirs(args.output_dir, exist_ok=True)
    log_eval = get_logger(args.output_dir, 'eval')

    # Data
    device = torch.device(args.device)
    data_loader = create_dataset(args)

    # Backend setup
    if args.backend == 'trt':
        if not args.engine:
            raise ValueError('--engine is required for trt backend')
        predictor = TRTInfer(args.engine)
        predict_fn = predictor
    elif args.backend == 'onnx':
        if not args.onnx:
            raise ValueError('--onnx is required for onnx backend')
        predictor = ONNXInfer(args.onnx, device=str(device))
        predict_fn = predictor
    else:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required for torch backend')
        model, _ = build_model(args)
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        missing = model.load_state_dict(state_dict, strict=False)
        if getattr(missing, 'missing_keys', None) or getattr(missing, 'unexpected_keys', None):
            print(f"State dict load info: {missing}")
        model.to(device).eval()
        predict_fn = model

    # Inference loop and save pngs
    save_root = args.output_dir
    infer_and_save_tensors(predict_fn, device, data_loader, save_root)

    # Compute metrics
    print("Computing metrics...")
    metrics = eval_metrics(log_eval, save_root, epoch=0)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Finished!")


if __name__ == '__main__':
    main()



