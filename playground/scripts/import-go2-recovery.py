"""Export FR-Net's Go2 recovery actor from a trusted local checkpoint.

Usage: python scripts/import-go2-recovery.py /path/to/model_2000.pt
Requires PyTorch, ONNX, and ONNX Runtime in the Python environment.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "public/policies/go2-recovery.onnx")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_state = {key[len("actor."):]: value for key, value in checkpoint["model_state_dict"].items()
                   if key.startswith("actor.")}
    expected_shapes = {"0.weight": (512, 45), "2.weight": (256, 512),
                       "4.weight": (128, 256), "6.weight": (12, 128)}
    for name, shape in expected_shapes.items():
        if tuple(actor_state[name].shape) != shape:
            raise ValueError(f"Unexpected {name} shape: {tuple(actor_state[name].shape)} != {shape}")
    actor = torch.nn.Sequential(
        torch.nn.Linear(45, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, 12),
    )
    actor.load_state_dict(actor_state)
    actor.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(actor, torch.zeros(1, 45), args.output, input_names=["observation"],
                      output_names=["action"], opset_version=17)
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    for _ in range(5):
        observation = rng.standard_normal((1, 45)).astype(np.float32)
        with torch.no_grad():
            expected = actor(torch.from_numpy(observation)).numpy()
        actual = session.run(["action"], {"observation": observation})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    print(f"Exported {args.output} from SHA256 {digest}; PyTorch/ONNX outputs match")


if __name__ == "__main__":
    main()
