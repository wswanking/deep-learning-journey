"""Auditable LoRA and a CPU packed uniform-int4 teaching alternative, not NF4 QLoRA."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

TARGET_NAMES = ("q_proj", "v_proj")
QUANTIZATION = "rowwise-symmetric-uniform-int4-not-nf4-v1"


class PackedInt4Linear(nn.Module):
    """Frozen 4-bit storage; unpack and compute in the incoming floating dtype.

    This deliberately simple implementation allocates a dequantized matrix at every
    forward. It demonstrates storage and gradient flow, not a fast low-bit kernel.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        scale = weight.abs().amax(dim=1, keepdim=True) / 7
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        codes = (torch.round(weight / scale).clamp(-7, 7).to(torch.int16) + 8).to(torch.uint8)
        flat = codes.flatten()
        if flat.numel() % 2:
            flat = F.pad(flat, (0, 1), value=8)
        packed = flat[0::2] | (flat[1::2] << 4)
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    def weight_in(self, dtype, device):
        packed = self.packed.to(device)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten()
        signed = codes[:self.in_features * self.out_features].to(dtype) - 8
        return signed.reshape(self.out_features, self.in_features) * self.scale.to(device=device, dtype=dtype)

    def forward(self, x):
        weight = self.weight_in(x.dtype, x.device)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if not isinstance(base, (nn.Linear, PackedInt4Linear)):
            raise TypeError("LoRA needs a declared linear base")
        if rank < 1 or rank > min(base.in_features, base.out_features):
            raise ValueError("rank must be positive and no greater than either matrix dimension")
        if alpha <= 0:
            raise ValueError("alpha must be positive in this teaching implementation")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        reference = base.weight if isinstance(base, nn.Linear) else base.scale
        self.lora_A = nn.Parameter(reference.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(reference.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** .5)

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)

    def merged_linear(self):
        reference = self.lora_A
        dense = nn.Linear(self.in_features, self.out_features,
                          bias=self.base.bias is not None,
                          device=reference.device, dtype=reference.dtype)
        base_weight = (self.base.weight if isinstance(self.base, nn.Linear)
                       else self.base.weight_in(reference.dtype, reference.device))
        with torch.no_grad():
            dense.weight.copy_(base_weight + self.scaling * self.lora_B @ self.lora_A)
            if dense.bias is not None:
                dense.bias.copy_(self.base.bias)
        return dense


def quantize_linears(model):
    """Replace all original Linear weights, keeping embeddings/norms floating."""
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, PackedInt4Linear(module))
            names.append(name)
    return names


def attach_lora(model, rank=4, alpha=8):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = []
    for index, block in enumerate(model.blocks):
        for leaf in TARGET_NAMES:
            parent = block.attention
            setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha))
            targets.append(f"blocks.{index}.attention.{leaf}")
    return targets


def adapter_state(model):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.endswith(("lora_A", "lora_B"))}


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    for name, buffer in model.named_buffers():
        digest.update(name.encode())
        digest.update(buffer.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def merged_copy(model):
    result = copy.deepcopy(model)
    for name, module in list(result.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, leaf = name.rpartition(".")
            setattr(result.get_submodule(parent_name), leaf, module.merged_linear())
    return result.eval()


def save_adapter(path, model, *, base_file_sha256, quantized, targets, protocol):
    first = next(module for module in model.modules() if isinstance(module, LoRALinear))
    payload = {
        "schema_version": 1, "type": "tiny-lm-lora-adapter",
        "base_file_sha256": base_file_sha256,
        "tokenizer": {"type": "utf8-byte", "vocab_size": 256, "special_tokens": None},
        "rank": first.rank, "alpha": first.alpha, "targets": targets,
        "quantization": QUANTIZATION if quantized else None,
        "compute_dtype": "float64", "adapter_state": adapter_state(model),
        "protocol": protocol,
    }
    torch.save(payload, path)


def load_adapter(path, base_checkpoint, load_base):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    actual_hash = hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest()
    if payload.get("schema_version") != 1 or payload.get("type") != "tiny-lm-lora-adapter":
        raise ValueError("unsupported adapter format")
    if payload.get("base_file_sha256") != actual_hash:
        raise ValueError("adapter requires the exact recorded base checkpoint")
    if payload.get("tokenizer") != {"type": "utf8-byte", "vocab_size": 256, "special_tokens": None}:
        raise ValueError("incompatible tokenizer protocol")
    if payload.get("compute_dtype") != "float64":
        raise ValueError("this example expects float64 compute")
    model, _ = load_base(base_checkpoint)
    model = model.double()
    if payload["quantization"] is not None:
        if payload["quantization"] != QUANTIZATION:
            raise ValueError("unsupported quantization method")
        quantize_linears(model)
    targets = attach_lora(model, payload["rank"], payload["alpha"])
    if targets != payload["targets"]:
        raise ValueError("adapter target modules differ")
    current = adapter_state(model)
    if current.keys() != payload["adapter_state"].keys():
        raise ValueError("adapter state keys differ")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in payload["adapter_state"].items():
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError("adapter tensor shape or dtype differs")
            parameters[name].copy_(value)
    return model.eval(), payload


def tensor_ledger(model, optimizer=None):
    """Persistent tensor payload, excluding activations, allocator metadata and RSS."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    nbytes = lambda value: value.numel() * value.element_size()
    return {
        "trainable_parameters": sum(p.numel() for p in trainable),
        "frozen_parameters": sum(p.numel() for p in frozen),
        "trainable_parameter_bytes": sum(nbytes(p) for p in trainable),
        "frozen_parameter_bytes": sum(nbytes(p) for p in frozen),
        "frozen_buffer_bytes": sum(nbytes(b) for b in model.buffers()),
        "gradient_bytes": sum(nbytes(p.grad) for p in trainable if p.grad is not None),
        "optimizer_tensor_bytes": sum(nbytes(v) for state in (optimizer.state.values() if optimizer else [])
                                     for v in state.values() if torch.is_tensor(v)),
        "scope": "persistent tensor payload only; excludes activations, dequantized temporary weights, Python, allocator, and RSS",
    }
