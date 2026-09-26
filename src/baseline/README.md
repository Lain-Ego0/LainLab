# Baseline policies

Versioned, directly playable policy artifacts live here. Each entry has a
`baseline.json` manifest plus the actual policy files:

```text
src/baseline/<robot>/<terrain>/
  baseline.json
  model.pt       # RSL-RL checkpoint for `uv run play --checkpoint-file`
  policy.onnx    # exported inference graph
```

Run a baseline directly in Viser:

```bash
uv run baseline-play opendoge flat --device cuda:0 --num-envs 1
uv run baseline-play opendoge rough --device cuda:0 --num-envs 1 --no-terminations True
uv run baseline-play opendoge getup --device cuda:0 --num-envs 1
```

Extra arguments after `<robot> <terrain>` are forwarded to mjlab's `play` CLI.
