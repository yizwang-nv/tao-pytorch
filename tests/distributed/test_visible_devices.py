# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in launch behavior must not change other model families."""

import io
import os
from types import SimpleNamespace

import pytest

from nvidia_tao_pytorch.core import entrypoint


@pytest.mark.parametrize(("network", "options", "binary", "visible"), [
    ("nvdinov2", {}, "python", "GPU-a,GPU-b"),
    ("rtdetr", {}, "torchrun", "0, 1"),
    ("dinov3", {"use_torchrun": True, "preserve_cuda": True}, "torchrun", "GPU-a,GPU-b"),
])
def test_launch_options_preserve_legacy_defaults(tmp_path, monkeypatch, network, options, binary, visible):
    """Only an opted-in caller preserves the scheduler mask under torchrun."""
    spec = tmp_path / "train.yaml"
    spec.write_text("train:\n  num_gpus: 2\n  gpu_ids: [0, 1]\n")
    calls = []

    def popen(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=io.StringIO(), wait=lambda: None, returncode=0)

    for key in ("WORLD_SIZE", "NODE_RANK", "RANK", "JOB_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    # Even an inherited DEFT flag cannot change another model's launch policy.
    monkeypatch.setenv("TAO_STRICT_MULTINODE", "1")
    monkeypatch.setattr(entrypoint.subprocess, "Popen", popen)
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", False)
    with pytest.raises(SystemExit) as result:
        entrypoint.launch({"subtask": "train", "experiment_spec_file": str(spec)}, [],
                          {"train": {"runner_path": "train.py"}}, network, **options)
    assert result.value.code == 0
    assert calls[0][0] == binary
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


@pytest.mark.parametrize("strict", [False, True])
def test_strict_validation_is_explicit_opt_in(tmp_path, monkeypatch, strict):
    """Legacy fallback remains unchanged; DINOv3 can fail closed."""
    spec = tmp_path / "train.yaml"
    spec.write_text("train:\n  num_gpus: 1\n  gpu_ids: [0]\n")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("TAO_STRICT_MULTINODE", "1")
    monkeypatch.delenv("JOB_ID", raising=False)
    monkeypatch.setattr(entrypoint, "TELEMETRY_AVAILABLE", False)
    monkeypatch.setattr(entrypoint.torch.cuda, "device_count", lambda: 1)

    def invalid(_logger):
        raise ValueError("bad rendezvous")

    monkeypatch.setattr(entrypoint, "validate_configs", invalid)
    monkeypatch.setattr(entrypoint.subprocess, "Popen", lambda *_a, **_k: SimpleNamespace(
        stdout=io.StringIO(), wait=lambda: None, returncode=0))
    with pytest.raises(RuntimeError if strict else SystemExit) as result:
        entrypoint.launch({"subtask": "train", "experiment_spec_file": str(spec)}, [],
                          {"train": {"runner_path": "train.py"}}, "dinov3" if strict else "nvdinov2",
                          strict_multinode=strict)
    if strict:
        assert "refusing to" in str(result.value)
    else:
        assert result.value.code == 0
