"""Reproducible YOLO-sem / U-Net location experiment for test1.ipynb.

Both candidates see the same 400x256 source crops, resized to 320x192, and
the same binary RLE-derived masks. This is a small practical comparison, not
an architecture benchmark or a four-class Kaggle submission.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 42
PATCH_W = 400
PATCH_H = 256
INPUT_W = 320
INPUT_H = 192
EPOCHS = 12
SAMPLE_COUNTS = {"train": 360, "validation": 90, "test": 120}
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")
CV_THREADS = 1
cv2.setNumThreads(CV_THREADS)


def paths(root):
    root = Path(root).resolve()
    return {"root": root, "dataset": root / "dataset", "models": root / "models",
            "results": root / "results"}


def _sample_manifest(manifest):
    parts = []
    for split, count in SAMPLE_COUNTS.items():
        candidates = manifest.loc[manifest.split.eq(split)]
        chosen = []
        for defect in (0, 1):
            pool = candidates.loc[candidates.has_defect.eq(defect)]
            n = round(count * len(pool) / len(candidates)) if defect else None
            if defect == 0:
                n = count - round(count * candidates.has_defect.mean())
            chosen.append(pool.sample(n=n, random_state=SEED + {"train": 0, "validation": 1, "test": 2}[split] + 10 * defect))
        parts.append(pd.concat(chosen).sort_values("ImageId"))
    return pd.concat(parts, ignore_index=True)


def _mask_for_image(rows, shape=(PATCH_H, PATCH_W * 4)):
    height, width = shape
    flat = np.zeros(height * width, dtype=np.uint8)
    for encoded in rows:
        pairs = np.fromstring(encoded, dtype=np.int64, sep=" ").reshape(-1, 2)
        for start, length in pairs:
            flat[int(start) - 1:int(start) - 1 + int(length)] = 1
    return flat.reshape((width, height)).T.copy()


def prepare(root, data_dir, manifest, labels):
    p = paths(root)
    index_path = p["dataset"] / "patch_index.csv"
    if index_path.exists():
        return pd.read_csv(index_path)
    p["dataset"].mkdir(parents=True, exist_ok=True)
    sampled = _sample_manifest(manifest)
    label_groups = labels.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
    records = []
    for split, group in sampled.groupby("split", sort=False):
        split_dir = "val" if split == "validation" else split
        for folder in ("images", "masks"):
            (p["dataset"] / folder / split_dir).mkdir(parents=True, exist_ok=True)
        for rec in group.itertuples(index=False):
            image = cv2.imread(str(Path(data_dir) / "train_images" / rec.ImageId), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (PATCH_H, PATCH_W * 4):
                raise ValueError(f"Unexpected image: {rec.ImageId}")
            truth = _mask_for_image(label_groups.get(rec.ImageId, []))
            for part in range(4):
                x0 = part * PATCH_W
                stem = f"{Path(rec.ImageId).stem}_p{part}"
                patch_image = cv2.resize(image[:, x0:x0 + PATCH_W], (INPUT_W, INPUT_H),
                                         interpolation=cv2.INTER_AREA)
                patch_mask = cv2.resize(truth[:, x0:x0 + PATCH_W], (INPUT_W, INPUT_H),
                                        interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(p["dataset"] / "images" / split_dir / f"{stem}.png"), patch_image)
                cv2.imwrite(str(p["dataset"] / "masks" / split_dir / f"{stem}.png"), patch_mask)
                records.append({"ImageId": rec.ImageId, "split": split, "part": part,
                                "stem": stem, "defect_pixels": int(patch_mask.sum()),
                                "original_defect": int(rec.has_defect)})
        print(f"Prepared {split}: {len(group)} original images / {len(group) * 4} patches", flush=True)
    index = pd.DataFrame(records)
    index.to_csv(index_path, index=False)
    config = {"seed": SEED, "epochs": EPOCHS, "samples": SAMPLE_COUNTS,
              "source_patch": [PATCH_W, PATCH_H], "input_patch": [INPUT_W, INPUT_H],
              "task": "binary semantic segmentation", "device": DEVICE,
              "models": {"YOLO": "yolo26n-sem.pt", "U-Net": "ResNet18 encoder U-Net"}}
    (p["dataset"] / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (p["dataset"] / "dataset.yaml").write_text(
        f"path: {p['dataset'].resolve()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nmasks_dir: masks\nnames:\n  0: background\n  1: defect\n", encoding="utf-8")
    return index


class PatchDataset(Dataset):
    def __init__(self, dataset_dir, index, split):
        self.root = Path(dataset_dir)
        self.rows = index.loc[index.split.eq(split)].reset_index(drop=True)
        self.folder = "val" if split == "validation" else split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem = self.rows.iloc[i].stem
        image = cv2.imread(str(self.root / "images" / self.folder / f"{stem}.png"),
                           cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root / "masks" / self.folder / f"{stem}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(stem)
        x = torch.from_numpy(image[None].astype(np.float32) / 255.0)
        y = torch.from_numpy(mask[None].astype(np.float32))
        return x, y


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Small grayscale U-Net; skips restore fine localization."""
    def __init__(self, base=16):
        super().__init__()
        self.down1 = DoubleConv(1, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.bridge = DoubleConv(base * 4, base * 8)
        self.up3 = DoubleConv(base * 12, base * 4)
        self.up2 = DoubleConv(base * 6, base * 2)
        self.up1 = DoubleConv(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(F.max_pool2d(a, 2))
        c = self.down3(F.max_pool2d(b, 2))
        d = self.bridge(F.max_pool2d(c, 2))
        x = self.up3(torch.cat([F.interpolate(d, size=c.shape[-2:], mode="bilinear", align_corners=False), c], 1))
        x = self.up2(torch.cat([F.interpolate(x, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1))
        x = self.up1(torch.cat([F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1))
        return self.head(x)


def _loss(logits, truth):
    bce = F.binary_cross_entropy_with_logits(logits, truth, pos_weight=torch.tensor(5., device=logits.device))
    prob = torch.sigmoid(logits)
    dice = (2 * (prob * truth).sum(dim=(1, 2, 3)) + 1) / (
        prob.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3)) + 1)
    return bce + (1 - dice).mean()


def train_unet(root, index):
    p = paths(root)
    p["models"].mkdir(parents=True, exist_ok=True)
    checkpoint = p["models"] / "unet_best.pt"
    history_path = p["results"] / "unet_history.csv"
    if checkpoint.exists() and history_path.exists():
        print("U-Net checkpoint reused", flush=True)
        return checkpoint
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = UNet().to(DEVICE)
    train_loader = DataLoader(PatchDataset(p["dataset"], index, "train"), batch_size=16,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(PatchDataset(p["dataset"], index, "validation"), batch_size=16,
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = []
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for x, y in val_loader:
                val_losses.append(float(_loss(model(x.to(DEVICE)), y.to(DEVICE)).item()))
        row = {"epoch": epoch, "train_loss": np.mean(train_losses),
               "val_loss": np.mean(val_losses), "elapsed_s": time.perf_counter() - started}
        history.append(row)
        print(f"U-Net {epoch}/{EPOCHS}: train {row['train_loss']:.4f}, val {row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(model.state_dict(), checkpoint)
    p["results"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return checkpoint


def train_yolo(root):
    from ultralytics import YOLO
    p = paths(root)
    checkpoint = p["models"] / "yolo" / "weights" / "best.pt"
    if checkpoint.exists():
        print("YOLO checkpoint reused", flush=True)
        return checkpoint
    p["models"].mkdir(parents=True, exist_ok=True)
    base_weights = p["root"].parent / "yolo26n-sem.pt"
    if not base_weights.exists():
        from urllib.request import urlretrieve
        urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-sem.pt",
                    base_weights)
    model = YOLO(str(base_weights))
    started = time.perf_counter()
    model.train(data=str(p["dataset"] / "dataset.yaml"), epochs=EPOCHS,
                imgsz=320, batch=16, device=DEVICE, workers=0, seed=SEED,
                project=str(p["models"]), name="yolo", exist_ok=True,
                plots=False, verbose=False, rect=True, mosaic=0.0, mixup=0.0,
                copy_paste=0.0, fliplr=0.0, flipud=0.0, degrees=0.0,
                translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                pretrained=True, save=True)
    print(f"YOLO training elapsed {time.perf_counter() - started:.1f}s", flush=True)
    return checkpoint


def train_models(root, index):
    # Separate processes are preferred in CLI so MPS memory is released.
    train_unet(root, index)
    train_yolo(root)


def _unet_model(root):
    p = paths(root)
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(p["models"] / "unet_best.pt", map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    return model


def _yolo_probabilities(model, gray_patches):
    """YOLO semantic logits -> defect probabilities at the shared patch size.

    These patches already have the training aspect ratio, so Ultralytics' rect
    predictor does not need letterbox padding. Upsampling then softmax matches
    its hard class map at a 0.5 threshold (checked on labeled patches).
    """
    x = torch.from_numpy(np.asarray(gray_patches).copy()).to(DEVICE)
    x = x[:, None].float().repeat(1, 3, 1, 1) / 255.0
    with torch.inference_mode():
        logits = model.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = F.interpolate(logits, size=(INPUT_H, INPUT_W),
                               mode="bilinear", align_corners=False)
        return F.softmax(logits, dim=1)[:, 1].cpu().numpy()


def predict_patch_scores(root, index, model_name, split):
    p = paths(root)
    p["results"].mkdir(parents=True, exist_ok=True)
    score_file = p["results"] / f"{model_name}_{split}_scores.npz"
    if score_file.exists():
        with np.load(score_file, allow_pickle=False) as saved:
            return saved["scores"], saved["stems"].astype(str)
    rows = index.loc[index.split.eq(split)]
    stems = rows.stem.to_numpy(dtype=str)
    folder = "val" if split == "validation" else split
    scores = np.zeros((len(stems), INPUT_H, INPUT_W), dtype=np.uint8)
    if model_name == "unet":
        model = _unet_model(root)
        loader = DataLoader(PatchDataset(p["dataset"], index, split), batch_size=16,
                            shuffle=False, num_workers=0)
        offset = 0
        with torch.inference_mode():
            for x, _ in loader:
                result = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()[:, 0]
                scores[offset:offset + len(result)] = np.round(result * 255).astype(np.uint8)
                offset += len(result)
    elif model_name == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(p["models"] / "yolo" / "weights" / "best.pt"))
        model.model.to(DEVICE).eval()
        for start in range(0, len(stems), 16):
            batch_stems = stems[start:start + 16]
            batch = np.stack([cv2.imread(str(p["dataset"] / "images" / folder / f"{stem}.png"),
                                         cv2.IMREAD_GRAYSCALE) for stem in batch_stems])
            probabilities = _yolo_probabilities(model, batch)
            scores[start:start + len(batch)] = np.round(probabilities * 255).astype(np.uint8)
            if start % 320 == 0:
                print(f"YOLO {split}: {min(start+16, len(stems))}/{len(stems)} patches", flush=True)
    else:
        raise ValueError(model_name)
    np.savez_compressed(score_file, scores=scores, stems=stems)
    return scores, stems


def _truth_masks(root, index, split):
    p = paths(root)
    rows = index.loc[index.split.eq(split)]
    folder = "val" if split == "validation" else split
    return np.stack([cv2.imread(str(p["dataset"] / "masks" / folder / f"{stem}.png"),
                                  cv2.IMREAD_GRAYSCALE) > 0 for stem in rows.stem])


def _metrics(scores, truth, index_split, cutoff):
    pred = scores >= cutoff
    hit = np.logical_and(pred, truth).sum()
    false_pixel = np.logical_and(pred, ~truth).sum()
    missed_pixel = np.logical_and(~pred, truth).sum()
    dice = 2 * hit / max(1, 2 * hit + false_pixel + missed_pixel)
    iou = hit / max(1, hit + false_pixel + missed_pixel)
    # Four patches are rejoined by original ImageId for image-level alert metrics.
    patch_truth = truth.reshape(len(truth), -1).any(axis=1)
    patch_pred = pred.reshape(len(pred), -1).any(axis=1)
    frame = pd.DataFrame({"ImageId": index_split.ImageId.to_numpy(),
                          "truth": patch_truth, "pred": patch_pred})
    image = frame.groupby("ImageId", sort=False)[["truth", "pred"]].any()
    tp = int((image.truth & image.pred).sum())
    fn = int((image.truth & ~image.pred).sum())
    fp = int((~image.truth & image.pred).sum())
    tn = int((~image.truth & ~image.pred).sum())
    return {"dice": float(dice), "iou": float(iou), "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "predicted_pixels": int(pred.sum()), "truth_pixels": int(truth.sum())}


def evaluate(root, index):
    p = paths(root)
    summary_path = p["results"] / "comparison.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    rows = []
    for model_name in ("unet", "yolo"):
        val_scores, val_stems = predict_patch_scores(root, index, model_name, "validation")
        test_scores, test_stems = predict_patch_scores(root, index, model_name, "test")
        val_index = index.loc[index.split.eq("validation")].reset_index(drop=True)
        test_index = index.loc[index.split.eq("test")].reset_index(drop=True)
        assert np.array_equal(val_stems, val_index.stem.to_numpy())
        assert np.array_equal(test_stems, test_index.stem.to_numpy())
        val_truth = _truth_masks(root, index, "validation")
        test_truth = _truth_masks(root, index, "test")
        choices = list(range(10, 241, 10))
        candidates = []
        for cutoff in choices:
            m = _metrics(val_scores, val_truth, val_index, cutoff)
            candidates.append((cutoff, m))
        feasible = [(cutoff, m) for cutoff, m in candidates if m["recall"] >= 0.95]
        if feasible:
            cutoff, val_metrics = max(feasible, key=lambda pair: (pair[1]["dice"], -pair[1]["fp"]))
        else:
            cutoff, val_metrics = max(candidates, key=lambda pair: (pair[1]["recall"], pair[1]["dice"]))
        test_metrics = _metrics(test_scores, test_truth, test_index, cutoff)
        row = {"model": model_name, "cutoff": cutoff / 255,
               "val_meets_95pct_recall": val_metrics["recall"] >= 0.95,
               **{f"val_{k}": v for k, v in val_metrics.items()},
               **{f"test_{k}": v for k, v in test_metrics.items()}}
        rows.append(row)
        print(f"{model_name}: val Dice {val_metrics['dice']:.3f}, recall {val_metrics['recall']:.3f}; "
              f"holdout Dice {test_metrics['dice']:.3f}, recall {test_metrics['recall']:.3f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(summary_path, index=False)
    return result


def reconstruct(root, index, model_name, image_id, split="test", cutoff=0.5):
    scores, stems = predict_patch_scores(root, index, model_name, split)
    rows = index.loc[index.split.eq(split)].reset_index(drop=True)
    assert np.array_equal(stems, rows.stem.to_numpy())
    indices = np.flatnonzero(rows.ImageId.eq(image_id).to_numpy())
    if len(indices) != 4:
        raise ValueError(image_id)
    patches = [cv2.resize((scores[i] >= round(cutoff * 255)).astype(np.uint8),
                          (PATCH_W, PATCH_H), interpolation=cv2.INTER_NEAREST)
               for i in indices]
    return np.concatenate(patches, axis=1).astype(bool)


def reconstruct_scores(root, index, model_name, image_id, split="test"):
    """Reassemble cached patch defect scores into a full-size uint8 image."""
    scores, stems = predict_patch_scores(root, index, model_name, split)
    rows = index.loc[index.split.eq(split)].reset_index(drop=True)
    assert np.array_equal(stems, rows.stem.to_numpy())
    indices = np.flatnonzero(rows.ImageId.eq(image_id).to_numpy())
    if len(indices) != 4:
        raise ValueError(image_id)
    patches = [cv2.resize(scores[i], (PATCH_W, PATCH_H), interpolation=cv2.INTER_NEAREST)
               for i in indices]
    return np.concatenate(patches, axis=1)


def compare_box_selection(root, index, labels, model_name, cutoff, top_k=2):
    """Compare top-k box rules against original RLE masks on held-out splits.

    A selected box counts all pixels inside its rectangle. Pixel precision,
    coverage, and Dice here describe box unions, not segmentation masks.
    """
    label_map = labels.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
    cutoff_byte = round(cutoff * 255)
    records = []
    for split in ("validation", "test"):
        rows = index.loc[index.split.eq(split)].reset_index(drop=True)
        patch_scores, stems = predict_patch_scores(root, index, model_name, split)
        if not np.array_equal(stems, rows.stem.to_numpy()):
            raise ValueError(f"Patch order mismatch: {split}")
        by_rule = {"mean": [], "sum": []}
        for image_id, ids in rows.groupby("ImageId", sort=False).indices.items():
            truth = _mask_for_image(label_map.get(image_id, [])).astype(bool)
            if not truth.any():
                continue
            scores = np.concatenate([
                cv2.resize(patch_scores[i], (PATCH_W, PATCH_H), interpolation=cv2.INTER_NEAREST)
                for i in ids], axis=1)
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (scores >= cutoff_byte).astype(np.uint8), connectivity=8)
            # Each connected component is a candidate rectangle. Score only
            # the predicted pixels inside that component, as in the display.
            score_sums = np.bincount(components.ravel(), weights=scores.ravel(), minlength=count)
            candidates = []
            for component in range(1, count):
                x, y, width, height, area = map(int, stats[component])
                candidates.append((x, y, width, height, area,
                                   score_sums[component] / area / 255.0))
            for rule in by_rule:
                key = (lambda box: box[5]) if rule == "mean" else (lambda box: box[4] * box[5])
                chosen = sorted(candidates, key=key, reverse=True)[:top_k]
                box_union = np.zeros_like(truth)
                for x, y, width, height, _, _ in chosen:
                    box_union[y:y + height, x:x + width] = True
                hit = int(np.logical_and(truth, box_union).sum())
                box_area = int(box_union.sum())
                truth_area = int(truth.sum())
                by_rule[rule].append((hit / max(1, box_area),
                                      hit / truth_area,
                                      2 * hit / max(1, box_area + truth_area),
                                      box_area))
        for rule, values in by_rule.items():
            average = np.asarray(values, dtype=float).mean(axis=0)
            records.append({"split": split, "rule": rule, "images": len(values),
                            "box_precision": average[0], "box_coverage": average[1],
                            "box_dice": average[2], "mean_box_area": average[3]})
    return pd.DataFrame(records)


def box_candidate_mask(scores, high_cutoff_byte, weak_area_trigger=2000,
                       low_cutoff_byte=60, adaptive=True):
    """Expand weak, high-confidence regions for box display only.

    Require at least one high-threshold seed. This prevents a previously
    empty image from acquiring boxes merely because of the lower cutoff.
    """
    high_mask = scores >= high_cutoff_byte
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        high_mask.astype(np.uint8), connectivity=8)
    largest_seed = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    expand = (adaptive and low_cutoff_byte < high_cutoff_byte
              and 0 < largest_seed < weak_area_trigger)
    cutoff_byte = low_cutoff_byte if expand else high_cutoff_byte
    return scores >= cutoff_byte, cutoff_byte, largest_seed


def compare_class3_box_expansion(root, index, labels, model_name, cutoff, top_k=2,
                                 weak_area_trigger=2000, low_cutoff_byte=60,
                                 rank_rule="sum"):
    """Compare seeded box expansion with the original cutoff on labeled images."""
    if rank_rule not in ("mean", "sum"):
        raise ValueError(rank_rule)
    all_labels = labels.groupby("ImageId")["EncodedPixels"].apply(list).to_dict()
    class3_labels = labels.loc[labels.ClassId.eq(3)].groupby("ImageId")[
        "EncodedPixels"].apply(list).to_dict()
    high_cutoff_byte = round(cutoff * 255)
    records = []
    for split in ("validation", "test"):
        rows = index.loc[index.split.eq(split)].reset_index(drop=True)
        patch_scores, stems = predict_patch_scores(root, index, model_name, split)
        if not np.array_equal(stems, rows.stem.to_numpy()):
            raise ValueError(f"Patch order mismatch: {split}")
        measures = {"baseline": {"dice": [], "class3_coverage": [],
                                 "normal_alert": [], "normal_area": []},
                    "adaptive": {"dice": [], "class3_coverage": [],
                                 "normal_alert": [], "normal_area": []}}
        for image_id, ids in rows.groupby("ImageId", sort=False).indices.items():
            scores = np.concatenate([
                cv2.resize(patch_scores[i], (PATCH_W, PATCH_H), interpolation=cv2.INTER_NEAREST)
                for i in ids], axis=1)
            truth = _mask_for_image(all_labels.get(image_id, [])).astype(bool)
            class3_truth = _mask_for_image(class3_labels.get(image_id, [])).astype(bool)
            for policy, values in measures.items():
                mask, _, _ = box_candidate_mask(
                    scores, high_cutoff_byte, weak_area_trigger, low_cutoff_byte,
                    adaptive=(policy == "adaptive"))
                count, components, stats, _ = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                score_sums = np.bincount(components.ravel(), weights=scores.ravel(), minlength=count)
                if rank_rule == "mean":
                    key = lambda component: score_sums[component] / stats[component, cv2.CC_STAT_AREA]
                else:
                    key = lambda component: score_sums[component]
                chosen = sorted(range(1, count), key=key, reverse=True)[:top_k]
                box_union = np.zeros_like(mask)
                for component in chosen:
                    x, y, width, height = map(int, stats[component, :4])
                    box_union[y:y + height, x:x + width] = True
                if truth.any():
                    hit = int(np.logical_and(truth, box_union).sum())
                    values["dice"].append(2 * hit / max(1, int(truth.sum() + box_union.sum())))
                else:
                    values["normal_alert"].append(int(box_union.any()))
                    values["normal_area"].append(int(box_union.sum()))
                if class3_truth.any():
                    values["class3_coverage"].append(
                        int(np.logical_and(class3_truth, box_union).sum()) / int(class3_truth.sum()))
        for policy, values in measures.items():
            records.append({"split": split, "policy": policy,
                            "defect_images": len(values["dice"]),
                            "class3_images": len(values["class3_coverage"]),
                            "box_dice": float(np.mean(values["dice"])),
                            "class3_coverage": float(np.mean(values["class3_coverage"])),
                            "normal_false_alarm_rate": float(np.mean(values["normal_alert"])),
                            "normal_box_area": float(np.mean(values["normal_area"]))})
    return pd.DataFrame(records)


def predict_original_scores(root, image_path, model_name):
    """Predict a full-size uint8 defect score map (0-255) for a new image."""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (PATCH_H, PATCH_W * 4):
        raise ValueError(f"Unexpected image: {image_path}")
    patches = [cv2.resize(gray[:, part * PATCH_W:(part + 1) * PATCH_W],
                          (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
               for part in range(4)]
    if model_name == "unet":
        model = _unet_model(root)
        x = torch.from_numpy(np.stack(patches)[:, None].astype(np.float32) / 255).to(DEVICE)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(x)).cpu().numpy()[:, 0]
    elif model_name == "yolo":
        from ultralytics import YOLO
        model = YOLO(str(paths(root)["models"] / "yolo" / "weights" / "best.pt"))
        model.model.to(DEVICE).eval()
        probabilities = _yolo_probabilities(model, np.stack(patches))
    else:
        raise ValueError(model_name)
    low_res = np.round(probabilities * 255).astype(np.uint8)
    score_patches = [cv2.resize(score, (PATCH_W, PATCH_H),
                                interpolation=cv2.INTER_NEAREST) for score in low_res]
    return np.concatenate(score_patches, axis=1)


def predict_original(root, image_path, model_name, cutoff=0.5):
    """Predict and reassemble a binary mask for one previously unseen image."""
    scores = predict_original_scores(root, image_path, model_name)
    return scores >= round(cutoff * 255)


def _sync():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


def benchmark(root, index, data_dir, n_images=24, rounds=3):
    p = paths(root)
    timing_path = p["results"] / "latency.csv"
    if timing_path.exists():
        return pd.read_csv(timing_path)
    ids = index.loc[index.split.eq("test"), "ImageId"].drop_duplicates().sample(
        n=n_images, random_state=SEED).to_list()
    unet = _unet_model(root)
    from ultralytics import YOLO
    yolo = YOLO(str(p["models"] / "yolo" / "weights" / "best.pt"))
    yolo.model.to(DEVICE).eval()
    measurements = []
    for model_name in ("unet", "yolo"):
        for sequence, image_id in enumerate(ids[:3] + ids * rounds):
            t0 = time.perf_counter()
            gray = cv2.imread(str(Path(data_dir) / "train_images" / image_id), cv2.IMREAD_GRAYSCALE)
            patches = [cv2.resize(gray[:, k*PATCH_W:(k+1)*PATCH_W],
                                  (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
                       for k in range(4)]
            if model_name == "unet":
                x = torch.from_numpy(np.stack(patches)[:, None].astype(np.float32) / 255).to(DEVICE)
                with torch.inference_mode():
                    masks = torch.sigmoid(unet(x)).cpu().numpy()[:, 0] >= 0.5
            else:
                masks = _yolo_probabilities(yolo, np.stack(patches)) >= 0.5
            _sync()
            # Include full-size mask stitching in end-to-end location latency.
            np.concatenate([cv2.resize(mask.astype(np.uint8), (PATCH_W, PATCH_H),
                                       interpolation=cv2.INTER_NEAREST) for mask in masks], axis=1)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if sequence >= 3:
                measurements.append({"model": model_name, "image_id": image_id,
                                     "latency_ms": elapsed_ms})
    frame = pd.DataFrame(measurements)
    frame.to_csv(timing_path, index=False)
    return frame


def chosen_model(comparison, latency):
    """Choose on validation only: 95% recall gate, then Dice; latency breaks near ties."""
    table = comparison.set_index("model")
    eligible = table.loc[table.val_meets_95pct_recall.astype(bool)]
    if len(eligible) == 0:
        return None, "두 모델 모두 검증 결함 재현율 95%에 미달해 채택을 보류합니다."
    if len(eligible) == 1:
        name = eligible.index[0]
        return name, f"{name}만 검증 결함 재현율 95%를 충족했습니다."
    order = eligible.sort_values("val_dice", ascending=False)
    if order.iloc[0].val_dice - order.iloc[1].val_dice >= 0.02:
        return order.index[0], "두 모델이 재현율 조건을 충족했고, 검증 마스크 Dice 차이가 0.02 이상입니다."
    speed = latency.groupby("model").latency_ms.median().sort_values()
    return speed.index[0], "검증 Dice 차이가 0.02 미만이어서 장당 중앙 처리 시간이 짧은 모델을 고릅니다."
