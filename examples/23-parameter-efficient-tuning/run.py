"""Chapter 23: common-base full FT, LoRA, and explicit uniform-int4 CPU alternative."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time

START = time.perf_counter()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from adapters import (LoRALinear, PackedInt4Linear, attach_lora, quantize_linears,
                      frozen_hash, merged_copy, save_adapter, load_adapter,
                      adapter_state, tensor_ledger, QUANTIZATION)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT / "examples/21-tiny-language-model"))
from model import load_checkpoint, inference_checkpoint
from tokenizer import encode, decode
spec = importlib.util.spec_from_file_location("chapter22_sft_data", ROOT / "examples/22-instruction-tuning/sft_data.py")
sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft)
IMPORT_SECONDS = time.perf_counter() - START


def loss_of(logits, labels):
    count = int((labels != -100).sum())
    if count < 1:
        raise ValueError("at least one supervised target is required")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction="sum") / count


def response_metrics(model, records):
    batch = sft.collate(records, model.config.context_length, "response_only")
    model.eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])
        selected = batch["labels"] != -100
        count = int(selected.sum())
        correct = int(((logits.argmax(-1) == batch["labels"]) & selected).sum())
        value = float(loss_of(logits, batch["labels"]))
    return {"mean_nll": value, "ppl_per_response_byte": math.exp(value),
            "teacher_forced_byte_accuracy": correct / count, "target_bytes": count}


def generation_metrics(model, records):
    model.eval()
    predictions = []
    with torch.no_grad():
        for record in records:
            ids = encode(sft.render_prompt(record))
            generated = []
            for _ in range(8):
                if len(ids) > model.config.context_length:
                    break
                token = int(model(torch.tensor([ids], dtype=torch.long))[0, -1].argmax())
                generated.append(token)
                ids.append(token)
                if token == 10:
                    break
            stopped = bool(generated and generated[-1] == 10)
            body = generated[:-1] if stopped else generated
            predictions.append({
                "id": record["id"], "task": record["task"], "instruction": record["instruction"],
                "expected": record["response"], "generated": decode(body),
                "generated_ids": generated, "stopped_on_lf": stopped,
                "exact_match": stopped and body == encode(record["response"]),
                "format_valid": stopped and len(body) == 2 and all(97 <= byte <= 102 for byte in body),
            })
    n = len(predictions)
    return {"exact_match": sum(p["exact_match"] for p in predictions) / n,
            "format_valid": sum(p["format_valid"] for p in predictions) / n,
            "count": n, "predictions": predictions}


def mechanics(check):
    w = torch.tensor([[1., 0., -1.], [0., 1., 1.]])
    a = torch.tensor([[1., -1., 0.]], requires_grad=True)
    b = torch.tensor([[2.], [-1.]], requires_grad=True)
    x = torch.tensor([2., -1., 1.], requires_grad=True)
    h = w @ x + b @ a @ x
    loss = .5 * ((h - torch.tensor([1., 1.])) ** 2).sum()
    loss.backward()
    check("hand_forward_and_gradients", torch.equal(h, torch.tensor([7., -3.])) and
          loss.item() == 26 and torch.equal(a.grad, torch.tensor([[32., -16., 16.]])) and
          torch.equal(b.grad, torch.tensor([[18.], [-12.]])) and
          torch.equal(x.grad, torch.tensor([22., -20., -10.])),
          h=h.detach().tolist(), loss=loss.item(), grad_a=a.grad.tolist(),
          grad_b=b.grad.tolist(), grad_x=x.grad.tolist())
    np_h = w.numpy() @ x.detach().numpy() + b.detach().numpy() @ a.detach().numpy() @ x.detach().numpy()
    check("numpy_torch_hand_forward", np.array_equal(np_h, h.detach().numpy()))

    layer = LoRALinear(nn.Linear(3, 2, bias=False), rank=1, alpha=1)
    with torch.no_grad():
        layer.base.weight.copy_(w)
        layer.lora_A.copy_(torch.tensor([[1., -1., 0.]]))
    inputs = torch.tensor([[2., -1., 1.]], requires_grad=True)
    check("zero_b_preserves_base", torch.equal(layer(inputs), layer.base(inputs)))
    (.5 * ((layer(inputs) - torch.tensor([[1., 1.]])) ** 2).sum()).backward()
    check("zero_b_first_gradient", layer.lora_A.grad.abs().max().item() == 0 and
          layer.lora_B.grad.norm().item() > 0 and layer.base.weight.grad is None)
    dead = copy.deepcopy(layer)
    with torch.no_grad():
        dead.lora_A.zero_()
        dead.lora_B.zero_()
    dead.zero_grad(set_to_none=True)
    dead(torch.tensor([[2., -1., 1.]])).sum().backward()
    check("both_zero_dead_branch", dead.lora_A.grad.abs().max().item() == 0 and
          dead.lora_B.grad.abs().max().item() == 0)
    with torch.no_grad():
        layer.lora_B.copy_(torch.tensor([[2.], [-1.]]))
    row = layer(inputs.detach())
    check("row_column_orientation_and_merge", torch.equal(row[0], h.detach()) and
          torch.equal(layer.merged_linear()(inputs.detach()), row))
    before = inputs.grad.clone()
    check("frozen_base_retains_input_gradient", layer.base.weight.grad is None and before.norm().item() > 0)

    linear = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[-1., -.3, .2, .9]]))
    quant = PackedInt4Linear(linear)
    restored = quant.weight_in(torch.float64, "cpu")
    error = (restored - linear.weight).abs().max().item()
    check("packed_uniform_int4_hand", quant.packed.tolist() == [97, 233] and
          torch.equal(restored, torch.tensor([[-1., -2/7, 1/7, 6/7]])) and
          error <= (quant.scale / 2).max().item() + 1e-15,
          packed=quant.packed.tolist(), reconstructed=restored.tolist(), max_error=error,
          scale=quant.scale.item())
    zeros = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        zeros.weight.zero_()
    qzero = PackedInt4Linear(zeros)
    check("zero_row_and_odd_packing", qzero.packed.numel() == 3 and
          bool((qzero.weight_in(torch.float64, "cpu") == 0).all()))
    with torch.no_grad():
        zeros.weight.copy_(torch.tensor([[0., 1., 2.], [-2., .1, .3]]))
    qodd = PackedInt4Linear(nn.Linear(3, 1, bias=False))
    check("odd_weight_count_restores_shape", qodd.packed.numel() == 2 and
          qodd.weight_in(torch.float64, "cpu").shape == (1, 3))
    xin = torch.ones(1, 4, requires_grad=True)
    quant(xin).sum().backward()
    check("quantized_frozen_input_gradient", torch.equal(xin.grad, restored) and
          list(quant.parameters()) == [])
    try:
        with torch.no_grad():
            broken = layer(inputs)
        broken.sum().backward()
        rejected = False
    except RuntimeError:
        rejected = True
    check("no_grad_breaks_adapter_training", rejected)
    return {"hand": {"h": h.detach().tolist(), "loss": loss.item()},
            "quantization_hand": {"scale": quant.scale.item(), "packed": quant.packed.tolist(),
                                  "restored": restored.tolist(), "max_error": error}}


def train_one(base, method, splits, args, folder, check):
    torch.manual_seed(2300)
    model = copy.deepcopy(base)
    quantized = method == "uniform_int4_lora"
    quantized_names = quantize_linears(model) if quantized else []
    targets = attach_lora(model, args.rank, args.alpha) if method != "full_ft" else []
    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad_(True)
    frozen_before = frozen_hash(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    check(method + "_parameter_count",
          sum(p.numel() for p in trainable) == (35328 if method == "full_ft" else 4 * args.rank * 64))
    optimizer = torch.optim.AdamW(trainable, lr=.0015, betas=(.9, .999), eps=1e-8, weight_decay=0)
    generator = torch.Generator().manual_seed(2201)
    batch_hash = hashlib.sha256()
    history = []
    best_score, best_step, best_state = math.inf, 0, None
    started = time.perf_counter()

    def record(step, minibatch_loss=None):
        nonlocal best_score, best_step, best_state
        train = response_metrics(model, splits["train"])
        validation = response_metrics(model, splits["validation"])
        history.append({"step": step, "train_nll": train["mean_nll"],
                        "validation_nll": validation["mean_nll"], "minibatch_loss": minibatch_loss})
        if validation["mean_nll"] < best_score:
            best_score, best_step = validation["mean_nll"], step
            best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    record(0)
    for step in range(1, args.updates + 1):
        indexes = torch.randint(len(splits["train"]), (16,), generator=generator)
        batch_hash.update(indexes.numpy().tobytes())
        records = [splits["train"][i] for i in indexes.tolist()]
        batch = sft.collate(records, model.config.context_length, "response_only")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        value = loss_of(logits, batch["labels"])
        value.backward()
        if step == 1:
            check(method + "_finite_gradients",
                  all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable))
            if method != "full_ft":
                check(method + "_frozen_grad_none",
                      all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer.step()
        if step % 50 == 0 or step == args.updates:
            record(step, float(value.detach()))
    ledger = tensor_ledger(model, optimizer)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(best_state[name])
    model.eval()
    check(method + "_frozen_tensors_unchanged", frozen_hash(model) == frozen_before)
    probe = sft.collate(splits["validation"], model.config.context_length)
    with torch.no_grad():
        expected = model(probe["input_ids"], probe["attention_mask"])
    if method != "full_ft":
        merged = merged_copy(model)
        with torch.no_grad():
            merge_error = (expected - merged(probe["input_ids"], probe["attention_mask"])).abs().max().item()
        check(method + "_merged_equivalence", merge_error < 1e-11, max_error=merge_error)
        filename = method + "-adapter.pt"
        save_adapter(folder / filename, model, base_file_sha256=args.base_hash,
                     quantized=quantized, targets=targets,
                     protocol={"chapter": 23, "objective": "response_only", "template": sft.TEMPLATE_VERSION,
                               "data_sha256": sft.fingerprints(), "best_step": best_step})
        restored, payload = load_adapter(folder / filename, args.base_checkpoint, load_checkpoint)
        check(method + "_adapter_only_payload", len(payload["adapter_state"]) == 8 and
              sum(v.numel() for v in payload["adapter_state"].values()) == 4 * args.rank * 64)
    else:
        filename = "full-ft.pt"
        torch.save(inference_checkpoint(model, {"chapter": 23, "objective": "response_only",
                                               "best_step": best_step}), folder / filename)
        restored, _ = load_checkpoint(folder / filename)
        restored = restored.double()
    with torch.no_grad():
        reload_error = (expected - restored(probe["input_ids"], probe["attention_mask"])).abs().max().item()
    check(method + "_checkpoint_reload", reload_error == 0, max_error=reload_error)
    return model, {
        "method": method, "best_step": best_step, "best_validation_nll": best_score,
        "history": history, "batch_order_sha256": batch_hash.hexdigest(),
        "targets": targets, "quantized_linear_names": quantized_names,
        "tensor_ledger": ledger, "checkpoint": str((folder / filename).relative_to(ROOT)),
        "checkpoint_file_bytes": (folder / filename).stat().st_size,
        "training_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "build/experiments/21-tiny-language-model/base-model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/experiments/23-parameter-efficient-tuning")
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.base_checkpoint = args.base_checkpoint.resolve()
    if args.updates < 1:
        raise ValueError("updates must be positive")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2300)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.base_hash = hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest()
    base, metadata = load_checkpoint(args.base_checkpoint)
    base = base.double().eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    checks = []

    def check(name, passed, **details):
        if not passed:
            raise AssertionError((name, details))
        checks.append({"name": name, "passed": True, **details})

    mechanics_result = mechanics(check)
    splits, data_metadata = sft.load_splits()
    check("shared_data_protocol", data_metadata["group_disjoint"] and
          data_metadata["sizes"] == {"train": 72, "validation": 24, "test": 24})
    reports = [{"method": "frozen_base", "best_step": 0, "tensor_ledger": tensor_ledger(base)}]
    models = [base]
    quant_baseline = copy.deepcopy(base)
    quantize_linears(quant_baseline)
    reports.append({"method": "uniform_int4_base", "best_step": 0, "tensor_ledger": tensor_ledger(quant_baseline)})
    models.append(quant_baseline)
    for method in ["full_ft", "lora", "uniform_int4_lora"]:
        model, report = train_one(base, method, splits, args, args.output_dir, check)
        models.append(model)
        reports.append(report)
    hashes = [r["batch_order_sha256"] for r in reports[2:]]
    check("identical_training_minibatch_orders", len(set(hashes)) == 1)
    check("original_checkpoint_file_unchanged",
          hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest() == args.base_hash)
    mismatch = torch.load(args.output_dir / "lora-adapter.pt", weights_only=True)
    mismatch["base_file_sha256"] = "0" * 64
    torch.save(mismatch, args.output_dir / "mismatched-base-adapter.pt")
    try:
        load_adapter(args.output_dir / "mismatched-base-adapter.pt", args.base_checkpoint, load_checkpoint)
        refused = False
    except ValueError:
        refused = True
    check("wrong_base_hash_refused", refused)
    # Training, hyperparameters and validation-selected checkpoints are now frozen.
    for model, report in zip(models, reports):
        report["metrics"] = {split: response_metrics(model, records) for split, records in splits.items()}
        report["test_generation"] = generation_metrics(model, splits["test"])
        report["train_generation"] = generation_metrics(model, splits["train"])
    result = {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "device": "cpu", "dtype": "float64", "threads": 1},
        "base": {"path": str(args.base_checkpoint), "file_sha256": args.base_hash,
                 "parameters": 35328, "metadata": metadata},
        "data": data_metadata, "data_sha256": sft.fingerprints(),
        "protocol": {"updates": args.updates, "batch": 16, "optimizer": "AdamW", "lr": .0015,
                     "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0,
                     "rank": args.rank, "alpha": args.alpha, "lora_dropout": 0,
                     "initialization_seed": 2300, "batch_seed": 2201, "validation_every": 50,
                     "selection": "minimum validation response-only mean NLL; test once after freeze",
                     "tokenizer": "fixed utf8-byte 0..255", "stop": "LF10 task rule, not global EOS",
                     "greedy_max_new_bytes": 8, "one_seed_only": True,
                     "compute": "float64 even for unpacked teaching int4 weights",
                     "quantization": QUANTIZATION, "nf4_qlora_executed": False},
        "mechanics": mechanics_result, "checks": checks, "runs": reports,
        "resources": {"seconds_including_import": time.perf_counter() - START,
                      "import_seconds": IMPORT_SECONDS, "process_peak_memory": "not measured",
                      "gpu_memory": "not measured; CPU only"},
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    if not args.skip_figures:
        from figures import write_figures
        write_figures(result, ROOT / "docs/assets/23-parameter-efficient-tuning")
    print(json.dumps({"checks_passed": len(checks), "runs": [
        {"method": r["method"], "best_step": r["best_step"], "parameters": r["tensor_ledger"]["trainable_parameters"],
         "test_nll": r["metrics"]["test"]["mean_nll"], "test_exact": r["test_generation"]["exact_match"],
         "train_exact": r["train_generation"]["exact_match"]} for r in reports],
         "resources": result["resources"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
