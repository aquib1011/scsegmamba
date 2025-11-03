"""
Export SCSEGAMBA model to ONNX and optionally build a TensorRT engine.

Examples:
  # ONNX only
  python export_to_onnx_trt.py \
    --checkpoint Resources_released/Resources_released/Checkpoints/DeepCrack/checkpoint_DeepCrack.pth \
    --onnx_path ./exports/scsegamba.onnx --opset 17 --height 512 --width 512

  # Build TensorRT engine on Jetson (requires TensorRT python)
  python export_to_onnx_trt.py \
    --checkpoint Resources_released/Resources_released/Checkpoints/DeepCrack/checkpoint_DeepCrack.pth \
    --onnx_path ./exports/scsegamba.onnx --engine_path ./exports/scsegamba_fp16.plan \
    --fp16 --height 512 --width 512
"""

import os
import argparse
import torch
import numpy as np

from models import build_model
from main import get_args_parser


def build_export_model(args: argparse.Namespace) -> torch.nn.Module:
    # Ensure correct runtime args for building
    args.phase = 'test'
    args.dataset_mode = 'crack'
    # Build model
    model, _ = build_model(args)
    # Load checkpoint
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        missing = model.load_state_dict(state_dict, strict=False)
        if getattr(missing, 'missing_keys', None) or getattr(missing, 'unexpected_keys', None):
            print(f"State dict load info: {missing}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.eval()
    model.to(args.device)
    return model


def export_onnx(model: torch.nn.Module, onnx_path: str, height: int, width: int, opset: int, dynamic: bool):
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    dummy = torch.randn(1, 3, height, width, device=next(model.parameters()).device)
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                        'output': {0: 'batch', 2: 'height', 3: 'width'}}
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    print(f"Saved ONNX to {onnx_path}")


def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool, int8: bool, max_workspace_size: int = 1 << 30):
    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError("TensorRT python package not available. Install TensorRT on Jetson.") from e

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 would require a calibrator; omitted here.

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Saved TensorRT engine to {engine_path}")


def parse_args():
    parent = argparse.ArgumentParser("Export to ONNX/TensorRT", parents=[get_args_parser()], conflict_handler='resolve')
    parent.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint .pth')
    parent.add_argument('--onnx_path', type=str, required=True, help='Output ONNX path')
    parent.add_argument('--engine_path', type=str, default=None, help='Output TensorRT engine path (.plan)')
    parent.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parent.add_argument('--height', type=int, default=512, help='Export input height')
    parent.add_argument('--width', type=int, default=512, help='Export input width')
    parent.add_argument('--dynamic', action='store_true', help='Export dynamic HxW and batch')
    parent.add_argument('--fp16', action='store_true', help='Enable FP16 for TensorRT engine')
    parent.add_argument('--int8', action='store_true', help='Enable INT8 for TensorRT engine (calibration not included)')
    return parent.parse_args()


def main():
    args = parse_args()
    # Force device selection for export
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_export_model(args)
    export_onnx(model, args.onnx_path, args.height, args.width, args.opset, args.dynamic)
    if args.engine_path is not None:
        build_trt_engine(args.onnx_path, args.engine_path, args.fp16, args.int8)


if __name__ == '__main__':
    main()



