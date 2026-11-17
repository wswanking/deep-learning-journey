"""CPU prefix-VLM curriculum, honest image interventions and exact restore."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import hashlib
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
from prefix_model import (ROOT, TinyPrefixVLM, load_language_checkpoint, vision_api,
                          file_sha256, state_sha256, save_vlm_checkpoint, load_vlm_checkpoint)
from data import make_data, collate, TASKS, prompt, answer
IMPORT_SECONDS = time.perf_counter() - START


def loss_of(logits, labels):
    count = int((labels != -100).sum())
    if count == 0:
        raise ValueError("at least one supervised answer byte required")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction="sum") / count


def evaluate_loss(model, records, bank, batch_size=64):
    model.eval()
    total, count, correct = 0., 0, 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = collate(records[start:start+batch_size], bank, model.lm.config.context_length)
            logits = model.forward_features(batch["features"], batch["input_ids"], batch["attention_mask"])
            selected = batch["labels"] != -100
            n = int(selected.sum())
            total += float(F.cross_entropy(logits.reshape(-1, 256), batch["labels"].reshape(-1),
                                          ignore_index=-100, reduction="sum"))
            count += n
            correct += int(((logits.argmax(-1) == batch["labels"]) & selected).sum())
    return {"mean_nll": total/count, "nll_sum": total, "answer_bytes": count,
            "teacher_forced_byte_accuracy": correct/count}


def generate_records(model, records, bank):
    """Batch each same-template question; never read gold answer during generation."""
    model.eval()
    output = {}
    with torch.no_grad():
        for task in TASKS:
            current = [record for record in records if record["task"] == task]
            if not current:
                continue
            for start in range(0, len(current), 48):
                rows = current[start:start+48]
                feats = bank[[record["image_index"] for record in rows]]
                ids = torch.tensor([list(prompt(task).encode("utf8"))] * len(rows), dtype=torch.long)
                valid = torch.ones_like(ids, dtype=torch.bool)
                active = torch.ones(len(rows), dtype=torch.bool)
                generated = [[] for _ in rows]
                for _ in range(20):
                    if feats.shape[1] + ids.shape[1] > model.lm.config.context_length:
                        break
                    logits = model.forward_features(feats, ids, valid)
                    last = feats.shape[1] + valid.sum(dim=1) - 1
                    next_ids = logits[torch.arange(len(rows)), last].argmax(dim=-1)
                    was_active = active.clone()
                    for i, token in enumerate(next_ids.tolist()):
                        if active[i]:
                            generated[i].append(token)
                    active &= next_ids != 10
                    if not bool(active.any()):
                        break
                    ids = torch.cat((ids, torch.where(was_active, next_ids, 0)[:, None]), dim=1)
                    valid = torch.cat((valid, was_active[:, None]), dim=1)
                allowed = {answer(k, task) for k in range(9)}
                for record, tokens in zip(rows, generated):
                    stopped = bool(tokens and tokens[-1] == 10)
                    body = tokens[:-1] if stopped else tokens
                    text = bytes(body).decode("utf8", errors="replace")
                    output[record["id"]] = {
                        "id": record["id"], "image_index": record["image_index"],
                        "class_id": record["class_id"], "task": task,
                        "expected": record["answer"], "generated": text, "generated_ids": tokens,
                        "stopped": stopped, "format_valid": stopped and text in allowed,
                        "exact_match": stopped and text == record["answer"],
                    }
    predictions = [output[row["id"]] for row in records]
    count = len(predictions)
    return {
        "exact_match": sum(p["exact_match"] for p in predictions)/count,
        "format_valid": sum(p["format_valid"] for p in predictions)/count,
        "count": count,
        "by_task": {task: sum(p["exact_match"] for p in predictions if p["task"] == task) /
                         sum(p["task"] == task for p in predictions) for task in TASKS},
        "predictions": predictions,
    }


def mechanics(model, bank, data, check):
    # A fully explicit linear bridge and its backwards map.
    v = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[1., 0.], [0., 1.], [1., -1.]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([.5, -.5, 1.], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[1., 0., -1.], [0., 2., 1.]], dtype=torch.float64)
    z = v @ w.T + b
    (z*g).sum().backward()
    check("linear_connector_hand_and_gradients", torch.equal(z, torch.tensor([[1.5, 1.5, 0.], [3.5, 3.5, 0.]], dtype=torch.float64)) and
          torch.equal(w.grad, torch.tensor([[1., 2.], [6., 8.], [2., 2.]], dtype=torch.float64)) and
          torch.equal(b.grad, torch.tensor([1., 2., 0.], dtype=torch.float64)) and
          torch.equal(v.grad, torch.tensor([[0., 1.], [1., 1.]], dtype=torch.float64)))
    np_z = v.detach().numpy() @ w.detach().numpy().T + b.detach().numpy()
    check("numpy_torch_connector", np.array_equal(np_z, z.detach().numpy()))
    batch = collate(data["train"]["records"][:6], bank["train"], model.lm.config.context_length)
    prefix_count = bank["train"].shape[1]
    for row, encoded in zip(batch["labels"], batch["encoded"]):
        first = prefix_count + encoded["prompt_bytes"] - 1
        check("first_answer_alignment_" + str(first) + "_" + str(encoded["answer_bytes"]),
              bool((row[:first] == -100).all()) and row[first] != -100 and
              int((row != -100).sum()) == encoded["answer_bytes"])
    check("visual_prefix_has_no_token_targets", bool((batch["labels"][:, :prefix_count] == -100).all()))
    ids = torch.tensor([[65, 66, 67, 68]])
    model.eval()
    with torch.no_grad():
        ordinary = model.lm(ids)
        without_image = model.forward_features(None, ids)
    check("zero_prefix_matches_shared_language_api", torch.equal(ordinary, without_image))
    features = bank["train"][:1]
    with torch.no_grad():
        original = model.forward_features(features, ids)
        changed = model.forward_features(features, torch.tensor([[65, 66, 120, 121]]))
    error = (original[:, :prefix_count+2] - changed[:, :prefix_count+2]).abs().max().item()
    check("answer_future_cannot_change_earlier_positions", error < 1e-6, max_error=error)
    padded = F.pad(ids, (0, 3))
    valid = torch.tensor([[True]*4+[False]*3])
    with torch.no_grad():
        longer = model.forward_features(features, padded, valid)
    padding_error = (longer[:, :prefix_count+4] - original).abs().max().item()
    check("right_padding_does_not_change_real_prefix", padding_error < 1e-5, max_error=padding_error, tolerance=1e-5, dtype="float32")
    precise = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        p0 = precise.forward_features(features.double(), ids)
        pp = precise.forward_features(features.double(), padded, valid)
    precise_error = (p0 - pp[:, :prefix_count+4]).abs().max().item()
    check("right_padding_float64_crosscheck", precise_error < 1e-11, max_error=precise_error, tolerance=1e-11)
    rejected = False
    try:
        model.forward_features(features, torch.ones(1, 49, dtype=torch.long))
    except ValueError:
        rejected = True
    check("visual_tokens_count_toward_context_limit", rejected)
    # Two visual positions + Q,: input + A input; targets A and LF only.
    p = torch.tensor([[[.25,.25,.25,.25]]*3+[[.05,.05,.8,.1],[.1,.2,.2,.5]]], dtype=torch.float64)
    logits = p.log().requires_grad_(True)
    labels = torch.tensor([[-100,-100,-100,2,3]])
    loss = loss_of(logits, labels)
    loss.backward()
    check("masked_answer_hand_and_prefix_gradient", abs(float(loss.detach())-(-math.log(.8)-math.log(.5))/2) < 1e-14 and
          float(logits.grad[:, :3].abs().max()) == 0)
    probe = copy.deepcopy(model)
    for parameter in probe.lm.parameters():
        parameter.requires_grad_(False)
    probe.train()
    out = probe.forward_features(batch["features"], batch["input_ids"], batch["attention_mask"])
    out.retain_grad()
    loss_of(out, batch["labels"]).backward()
    check("frozen_language_still_trains_connector", all(p.grad is None for p in probe.lm.parameters()) and
          all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in probe.connector.parameters()) and
          sum(float(p.grad.abs().sum()) for p in probe.connector.parameters()) > 0)
    check("ignored_logits_zero_local_gradient", float(out.grad[batch["labels"] == -100].abs().max()) == 0)
    return {"linear_bridge_output": z.detach().tolist(), "hand_answer_mean_nll": float(loss.detach())}


def train_stage(initial, stage, records, validation, banks, args, check):
    model = copy.deepcopy(initial)
    for parameter in model.lm.parameters():
        parameter.requires_grad_(stage != "connector_only")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    expected = 4192 if stage == "connector_only" else 4192+35328
    check(stage+"_trainable_count", sum(p.numel() for p in trainable) == expected)
    frozen_language = state_sha256(model.lm)
    optimizer = torch.optim.AdamW(trainable, lr=.01 if stage == "connector_only" else .002,
                                 betas=(.9,.999), eps=1e-8, weight_decay=0)
    updates = args.connector_updates if stage == "connector_only" else args.joint_updates
    generator = torch.Generator().manual_seed(2904)
    history, order = [], hashlib.sha256()
    best, best_step, best_state = math.inf, 0, None
    started = time.perf_counter()

    def record(step):
        nonlocal best, best_step, best_state
        training = evaluate_loss(model, records, banks["train"])
        valid = evaluate_loss(model, validation, banks["validation"])
        history.append({"step": step, "train_mean_nll": training["mean_nll"], "validation_mean_nll": valid["mean_nll"]})
        if valid["mean_nll"] < best:
            best, best_step = valid["mean_nll"], step
            best_state = copy.deepcopy(model.state_dict())

    record(0)
    for step in range(1, updates+1):
        indexes = torch.randint(len(records), (16,), generator=generator)
        order.update(indexes.numpy().tobytes())
        batch = collate([records[index] for index in indexes.tolist()], banks["train"], model.lm.config.context_length)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_features(batch["features"], batch["input_ids"], batch["attention_mask"])
        loss = loss_of(logits, batch["labels"])
        loss.backward()
        if step == 1:
            check(stage+"_finite_gradient", all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable))
        optimizer.step()
        if step % 50 == 0 or step == updates:
            record(step)
    model.load_state_dict(best_state)
    model.eval()
    if stage == "connector_only":
        check("stage1_language_frozen_bitexact", state_sha256(model.lm) == frozen_language)
    filename = {"connector_only": "connector-only.pt", "joint": "joint.pt", "text_prior": "text-prior.pt"}[stage]
    save_vlm_checkpoint(args.output_dir/filename, model, base_lm_checkpoint=args.language_checkpoint,
                        vision_checkpoint=args.vision_checkpoint,
                        protocol={"chapter":29,"stage":stage,"best_step":best_step,
                                  "template":"ascii-three-questions-v1","visual_tokens":16,
                                  "dtype":"float32","stop_byte":10,"max_new_bytes":20,
                                  "image_condition":"constant real black-image features" if stage=="text_prior" else "true image features",
                                  "data_metadata":args.data_metadata})
    restored, _ = load_vlm_checkpoint(args.output_dir/filename, args.language_checkpoint, args.vision_checkpoint)
    sample = collate(validation[:12], banks["validation"], model.lm.config.context_length)
    with torch.no_grad():
        before = model.forward_features(sample["features"], sample["input_ids"], sample["attention_mask"])
        after = restored.forward_features(sample["features"], sample["input_ids"], sample["attention_mask"])
    check(stage+"_checkpoint_exact", torch.equal(before, after))
    return model, {"stage":stage,"trainable_parameters":expected,"updates":updates,"best_step":best_step,
                   "best_validation_nll":best,"history":history,"batch_sha256":order.hexdigest(),
                   "checkpoint":str(args.output_dir/filename),"training_seconds":time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-checkpoint",type=Path,default=ROOT/"build/experiments/21-tiny-language-model/base-model.pt")
    parser.add_argument("--vision-checkpoint",type=Path,default=ROOT/"build/experiments/28-multimodal-representations/dual-encoder.pt")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"build/experiments/29-vision-language-models")
    parser.add_argument("--connector-updates",type=int,default=400)
    parser.add_argument("--joint-updates",type=int,default=800)
    parser.add_argument("--skip-figures",action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.language_checkpoint = args.language_checkpoint.resolve()
    args.vision_checkpoint = args.vision_checkpoint.resolve()
    if min(args.connector_updates,args.joint_updates)<1:
        raise ValueError("positive updates required")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(2900)
    torch.use_deterministic_algorithms(True)
    checks = []

    def check(name, passed, **details):
        if not passed:
            raise AssertionError((name,details))
        checks.append({"name":name,"passed":True,**details})

    language_hash = file_sha256(args.language_checkpoint)
    vision_hash = file_sha256(args.vision_checkpoint)
    lm, lm_metadata = load_language_checkpoint(args.language_checkpoint)
    dual, vision_metadata = vision_api.load_checkpoint(args.vision_checkpoint)
    vision = dual.vision.eval()
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    before_vision = state_sha256(vision)
    data, args.data_metadata = make_data()
    check("image_group_split_and_upstream_hashes", args.data_metadata["image_disjoint"] and args.data_metadata["no_exact_upstream_image_overlap"])
    banks = {}
    with torch.no_grad():
        for split, item in data.items():
            banks[split] = vision.forward_tokens(item["images"]).detach()
        black = vision.forward_tokens(torch.zeros(1,3,16,16)).detach()
    check("shared_visual_token_shape", banks["train"].shape == (288,16,32))
    model = TinyPrefixVLM(lm)
    check("connector_parameter_count",sum(p.numel() for p in model.connector.parameters())==4192)
    mechanics_result = mechanics(model,banks,data,check)
    describe = lambda rows:[r for r in rows if r["task"]=="describe"]
    connector, stage1 = train_stage(model,"connector_only",describe(data["train"]["records"]),
                                     describe(data["validation"]["records"]),banks,args,check)
    joint, stage2 = train_stage(connector,"joint",data["train"]["records"],data["validation"]["records"],banks,args,check)
    blank_banks = {split:black.repeat(len(item["images"]),1,1) for split,item in data.items()}
    prior, stage3 = train_stage(connector,"text_prior",data["train"]["records"],data["validation"]["records"],blank_banks,args,check)
    check("joint_and_prior_identical_minibatch_order",stage2["batch_sha256"]==stage3["batch_sha256"])
    check("frozen_vision_unchanged",state_sha256(vision)==before_vision and all(p.grad is None for p in vision.parameters()))
    check("upstream_checkpoints_unchanged",file_sha256(args.language_checkpoint)==language_hash and file_sha256(args.vision_checkpoint)==vision_hash)
    # All choices/checkpoints are frozen before downstream test and interventions.
    for trained, report, features in [(connector,stage1,banks),(joint,stage2,banks),(prior,stage3,blank_banks)]:
        report["loss"] = {split:evaluate_loss(trained,item["records"],features[split]) for split,item in data.items()}
        report["test_generation"] = generate_records(trained,data["test"]["records"],features["test"])
    test_rows = data["test"]["records"]
    views = args.data_metadata["splits"]["test"]["views_per_class"]
    perm = torch.tensor([((index//views+4)%9)*views+index%views for index in range(len(data["test"]["images"]))])
    shuffled = generate_records(joint,test_rows,banks["test"][perm])
    for prediction in shuffled["predictions"]:
        alternate_class = int(data["test"]["labels"][perm[prediction["image_index"]]])
        prediction["counterfactual_expected"] = answer(alternate_class,prediction["task"])
        prediction["matches_counterfactual_image"] = prediction["stopped"] and prediction["generated"] == prediction["counterfactual_expected"]
    shuffled["counterfactual_image_accuracy"] = sum(p["matches_counterfactual_image"] for p in shuffled["predictions"])/len(shuffled["predictions"])
    blanked = generate_records(joint,test_rows,blank_banks["test"])
    prior_predictions = stage3["test_generation"]["predictions"]
    check("text_prior_constant_within_question",all(len({p["generated"] for p in prior_predictions if p["task"]==task})==1 for task in TASKS))
    for split, item in data.items():
        (args.output_dir/(split+"-records.jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in item["records"]),encoding="utf8")
    result = {
        "run_at":datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,
                       "device":"cpu","dtype":"float32","threads":1},
        "bases":{"language_file_sha256":language_hash,"vision_file_sha256":vision_hash,
                 "language_metadata":lm_metadata,"vision_metadata":vision_metadata,
                 "language_parameters":sum(p.numel() for p in lm.parameters()),
                 "vision_parameters_including_unused_image_projection":sum(p.numel() for p in vision.parameters())},
        "data":args.data_metadata,"mechanics":mechanics_result,"checks":checks,"stages":[stage1,stage2,stage3],
        "protocol":{"stage1":"describe only; frozen vision and LM; train connector",
                    "stage2":"three questions; frozen vision; train connector and LM",
                    "control":"same stage1 starting state and stage2 order/budget; constant true black-image features",
                    "batch":16,"initialization_seed":2900,"sampling_seed":2904,"validation_every":50,
                    "selection":"minimum answer-byte-weighted validation NLL; freeze then test",
                    "optimizer":"AdamW","stage1_lr":.01,"stage2_lr":.002,"betas":[.9,.999],"eps":1e-8,"weight_decay":0,
                    "greedy_max_new_bytes":20,"stop_byte":10,"context_length":64,"visual_prefix_length":16,
                    "one_seed_only":True,"cache":"frozen eval vision features in memory; connector remains in graph",
                    "real_vision_model_or_llava_reproduction":False},
        "interventions":{"permutation_rule":"class +4 modulo9, preserve view index; both color and shape change",
                         "shuffled":shuffled,"blank_image":blanked,
                         "text_only_balanced_exact_ceiling":7/27},
        "resources":{"seconds_including_import":time.perf_counter()-START,"import_seconds":IMPORT_SECONDS,
                     "feature_cache_bytes":sum(v.numel()*v.element_size() for v in banks.values()),
                     "peak_rss":"not measured","gpu_memory":"not measured; CPU only"},
    }
    (args.output_dir/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    if not args.skip_figures:
        from figures import write_figures
        write_figures(result,ROOT/"docs/assets/29-vision-language-models")
    print(json.dumps({"checks_passed":len(checks),"stages":[{"stage":r["stage"],"best_step":r["best_step"],
          "test_nll":r["loss"]["test"]["mean_nll"],"test_exact":r["test_generation"]["exact_match"],
          "by_task":r["test_generation"]["by_task"]} for r in result["stages"]],
          "shuffle_original_accuracy":shuffled["exact_match"],"shuffle_counter_image_accuracy":shuffled["counterfactual_image_accuracy"],
          "blank_image_accuracy":blanked["exact_match"],"resources":result["resources"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
