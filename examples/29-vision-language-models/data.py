"""Original procedural multimodal questions; split by image before creating tasks."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from prefix_model import vision_data

COLORS = ["red", "green", "blue"]
SHAPES = ["circle", "square", "cross"]
TASKS = ["color", "shape", "describe"]


def prompt(task):
    if task not in TASKS:
        raise ValueError("unknown question task")
    return task + ":\n"


def answer(class_id, task):
    color, shape = divmod(int(class_id), 3)
    if task == "color":
        return COLORS[color]
    if task == "shape":
        return SHAPES[shape]
    if task == "describe":
        return COLORS[color] + " " + SHAPES[shape]
    raise ValueError("unknown task")


def make_data():
    specs = {"train": (2901, 32), "validation": (2902, 8), "test": (2903, 12)}
    result, metadata, pixel_hashes = {}, {}, {}
    for split, (seed, views) in specs.items():
        images, labels = vision_data.make_split(seed, views)
        records = []
        for index, class_id in enumerate(labels.tolist()):
            for task in TASKS:
                records.append({"id": f"{split}:{index}:{task}", "image_id": f"{split}:{index}",
                                "image_index": index, "class_id": class_id, "task": task,
                                "prompt": prompt(task), "answer": answer(class_id, task)})
        result[split] = {"images": images, "labels": labels, "records": records}
        pixel_hashes[split] = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in images}
        raw = images.numpy().tobytes() + labels.numpy().tobytes()
        metadata[split] = {"seed": seed, "views_per_class": views, "images": len(labels),
                           "records": len(records), "tensor_sha256": hashlib.sha256(raw).hexdigest(),
                           "tensor_bytes": len(raw)}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if pixel_hashes[a] & pixel_hashes[b]:
            raise AssertionError("exact image overlap across downstream splits")
    # New rendering seeds, same concepts as the trained vision encoder.
    upstream, _ = vision_data.make_data()
    upstream_hashes = {hashlib.sha256(row.numpy().tobytes()).hexdigest()
                       for images, _ in upstream.values() for row in images}
    if any(hashes & upstream_hashes for hashes in pixel_hashes.values()):
        raise AssertionError("downstream exact image overlaps the vision-pretraining images")
    return result, {
        "generator": "chapter28-rgb-shapes-v1", "language_template": "ascii-three-questions-v1",
        "colors": COLORS, "shapes": SHAPES, "tasks": TASKS, "splits": metadata,
        "image_disjoint": True, "no_exact_upstream_image_overlap": True,
        "license": "original procedural data; repository license",
        "scope": "same nine concepts, new rendering seeds; not unseen classes or natural images",
    }


def collate(records, feature_bank, context_length=64):
    if not records:
        raise ValueError("nonempty batch required")
    encoded = []
    for row in records:
        prefix = list(row["prompt"].encode("utf8"))
        target = list((row["answer"] + "\n").encode("utf8"))
        full = prefix + target
        if not prefix or len(full) - 1 + feature_bank.shape[1] > context_length:
            raise ValueError("empty question or multimodal context overflow")
        labels = full[1:]
        labels[:len(prefix) - 1] = [-100] * (len(prefix) - 1)
        encoded.append({"input_ids": full[:-1], "labels": labels,
                        "prompt_bytes": len(prefix), "answer_bytes": len(target)})
    length = max(len(row["input_ids"]) for row in encoded)
    ids = torch.zeros(len(records), length, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full((len(records), feature_bank.shape[1] + length), -100, dtype=torch.long)
    for index, row in enumerate(encoded):
        count = len(row["input_ids"])
        ids[index, :count] = torch.tensor(row["input_ids"])
        valid[index, :count] = True
        labels[index, feature_bank.shape[1]:feature_bank.shape[1] + count] = torch.tensor(row["labels"])
    features = feature_bank[[row["image_index"] for row in records]]
    return {"features": features, "input_ids": ids, "attention_mask": valid,
            "labels": labels, "encoded": encoded}
