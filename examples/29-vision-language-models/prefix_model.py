"""Chapter-29 continuous visual prefix wrapper; chapter 21/28 sources stay read-only."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT / "examples/21-tiny-language-model"))
from model import load_checkpoint as load_language_checkpoint

def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

vision_api = import_source("chapter28_vision_api", ROOT / "examples/28-multimodal-representations/model.py")
vision_data = import_source("chapter28_vision_data", ROOT / "examples/28-multimodal-representations/data.py")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class TinyPrefixVLM(nn.Module):
    """Return [B,P+T,256] logits, visual positions are not vocabulary IDs."""
    def __init__(self, language_model, visual_width=32):
        super().__init__()
        self.lm = language_model
        self.visual_width = visual_width
        width = language_model.config.d_model
        self.connector = nn.Sequential(nn.Linear(visual_width, 64), nn.GELU(), nn.Linear(64, width))
        reference = next(language_model.parameters())
        self.connector.to(dtype=reference.dtype, device=reference.device)
        for module in self.connector.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
                nn.init.zeros_(module.bias)

    def forward_features(self, features, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must have [B,T] long shape")
        batch, length = input_ids.shape
        if length < 1:
            raise ValueError("nonempty text prefix required")
        valid = (torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None
                 else attention_mask.to(device=input_ids.device, dtype=torch.bool))
        if valid.shape != input_ids.shape or not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("each text input must have a nonempty real prefix then right padding")
        if features is None:
            prefix = self.lm.token_embedding.weight.new_empty(batch, 0, self.lm.config.d_model)
        else:
            if features.ndim != 3 or features.shape[0] != batch or features.shape[-1] != self.visual_width:
                raise ValueError("visual features must have [B,P,visual_width] shape")
            prefix = self.connector(features.to(dtype=self.lm.token_embedding.weight.dtype))
        count = prefix.shape[1]
        if count + length > self.lm.config.context_length:
            raise ValueError("visual prefix plus text input exceeds language context; no silent truncation")
        text = self.lm.token_embedding(input_ids)
        hidden = torch.cat((prefix, text), dim=1)
        positions = torch.arange(count + length, device=input_ids.device)
        hidden = self.lm.embedding_dropout(hidden + self.lm.position_embedding(positions)[None])
        combined_valid = torch.cat((torch.ones(batch, count, dtype=torch.bool, device=input_ids.device), valid), dim=1)
        for block in self.lm.blocks:
            hidden = block(hidden, combined_valid)
        return self.lm.lm_head(self.lm.final_norm(hidden))


def save_vlm_checkpoint(path, model, *, base_lm_checkpoint, vision_checkpoint, protocol):
    torch.save({
        "schema_version": 1, "type": "tiny-continuous-prefix-vlm",
        "base_lm_sha256": file_sha256(base_lm_checkpoint),
        "vision_sha256": file_sha256(vision_checkpoint), "visual_width": model.visual_width,
        "language_config": dict(vars(model.lm.config)),
        "tokenizer": {"type": "utf8-byte", "vocab_size": 256, "special_tokens": None},
        "model_state": model.state_dict(), "protocol": protocol,
    }, path)


def load_vlm_checkpoint(path, base_lm_checkpoint, vision_checkpoint):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("type") != "tiny-continuous-prefix-vlm":
        raise ValueError("unsupported prefix model checkpoint")
    if payload["base_lm_sha256"] != file_sha256(base_lm_checkpoint) or payload["vision_sha256"] != file_sha256(vision_checkpoint):
        raise ValueError("exact recorded language/vision base checkpoints are required")
    if payload["tokenizer"] != {"type": "utf8-byte", "vocab_size": 256, "special_tokens": None}:
        raise ValueError("incompatible tokenizer")
    lm, _ = load_language_checkpoint(base_lm_checkpoint)
    lm = lm.to(dtype=payload["model_state"]["lm.token_embedding.weight"].dtype)
    if dict(vars(lm.config)) != payload["language_config"]:
        raise ValueError("language configuration differs")
    model = TinyPrefixVLM(lm, payload["visual_width"])
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), {k: v for k, v in payload.items() if k != "model_state"}
