"""Convert the trusted G1 DWAQ phase TorchScript policy to browser ONNX.

Usage: python scripts/import-g1-dwaq.py /path/to/policy/g1_dwaq_phase/policy.pt
Requires PyTorch, ONNX, ONNX Runtime and NumPy. No source checkout at runtime.
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
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "public/policies/g1-dwaq-phase.onnx")
    args = parser.parse_args()
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    sample = torch.zeros(1, 500)
    with torch.inference_mode():
        output = policy(sample)
    if tuple(output.shape) != (1, 29):
        raise ValueError("Expected a 500-input, 29-output DWAQ phase policy")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(policy, sample, str(args.output), input_names=["observation"],
                      output_names=["action"], opset_version=17)
    onnx.checker.check_model(str(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(17)
    samples = [np.zeros((1, 500), dtype=np.float32)]
    for scale in [.1, 1., 3.]:
        samples.extend([rng.normal(0, scale, (1, 500)).astype(np.float32) for _ in range(5)])
    max_error = 0.
    for observation in samples:
        with torch.inference_mode():
            expected = policy(torch.from_numpy(observation)).numpy()
        actual = session.run(["action"], {"observation": observation})[0]
        if not np.isfinite(actual).all():
            raise ValueError("ONNX returned non-finite actions")
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    print("Exported {}\nSource SHA256: {}\n{} inference comparisons passed; max absolute error {:.8g}".format(
        args.output, hashlib.sha256(args.policy.read_bytes()).hexdigest(), len(samples), max_error))


if __name__ == "__main__":
    main()
