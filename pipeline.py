from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from tqdm import tqdm

from .data import feature_loader, training_loaders
from .feature_guard import FeatureGuard
from .model import WFRNet
from .utils import image_files, sample_id, save_json, seed_everything
from .watermark import embed, verify


def protect_dataset(
    input_root: str | Path,
    output_root: str | Path,
    key: str,
    repetitions: int,
) -> Path:
    source = Path(input_root)
    destination = Path(output_root)
    records: list[dict[str, object]] = []
    classes = sorted(path.name for path in source.iterdir() if path.is_dir())

    for path in tqdm(image_files(source), desc="Protecting images"):
        relative = path.relative_to(source)
        class_name = relative.parts[0]
        identifier = sample_id(relative.as_posix())
        output_path = destination / class_name / f"{identifier}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            protected = embed(image, key, identifier, class_name, repetitions)
            protected.save(output_path, format="PNG")
        records.append(
            {
                "sample_id": identifier,
                "source_path": relative.as_posix(),
                "file_name": output_path.name,
                "true_class": class_name,
                "observed_class": class_name,
                "class_index": classes.index(class_name),
                "poisoned": False,
                "attack": "none",
            }
        )

    manifest_path = destination / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    return manifest_path


def poison_dataset(
    input_root: str | Path,
    output_root: str | Path,
    attack: str,
    ratio: float,
    seed: int,
    source_class: str | None = None,
    target_class: str | None = None,
) -> Path:
    source = Path(input_root)
    destination = Path(output_root)
    manifest = pd.read_csv(source / "manifest.csv")
    rng = np.random.default_rng(seed)
    candidates = manifest.index.to_numpy()
    if source_class:
        candidates = manifest.index[manifest["true_class"] == source_class].to_numpy()
    count = max(1, int(len(candidates) * ratio))
    selected = set(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
    classes = sorted(manifest["true_class"].unique())

    updated_records = []
    for index, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Applying attack"):
        observed_class = str(row["true_class"])
        is_poisoned = index in selected
        input_path = source / observed_class / str(row["file_name"])
        image = Image.open(input_path).convert("RGB")

        if is_poisoned and attack == "label_flip":
            if target_class:
                observed_class = target_class
            else:
                alternatives = [name for name in classes if name != observed_class]
                observed_class = str(rng.choice(alternatives))
        elif is_poisoned and attack == "gaussian_noise":
            array = np.asarray(image, dtype=np.float32)
            noise = rng.normal(0, 18, array.shape)
            image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        elif is_poisoned and attack == "patch_trigger":
            array = np.asarray(image, dtype=np.uint8).copy()
            size = max(3, min(array.shape[:2]) // 12)
            array[-size:, -size:, :] = np.array([255, 255, 255], dtype=np.uint8)
            image = Image.fromarray(array)

        output_path = destination / observed_class / str(row["file_name"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        updated = row.to_dict()
        updated["observed_class"] = observed_class
        updated["poisoned"] = is_poisoned
        updated["attack"] = attack if is_poisoned else "none"
        updated_records.append(updated)

    output_manifest = destination / "manifest.csv"
    pd.DataFrame(updated_records).to_csv(output_manifest, index=False)
    return output_manifest


def train_model(data_root: str | Path, output_path: str | Path, config: dict) -> dict:
    seed = int(config["seed"])
    seed_everything(seed)
    data_config = config["data"]
    train_config = config["training"]
    train_loader, validation_loader, classes = training_loaders(
        data_root,
        int(data_config["image_size"]),
        int(data_config["batch_size"]),
        int(data_config["workers"]),
        float(train_config["validation_split"]),
        seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WFRNet(len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history = []

    for epoch in range(int(train_config["epochs"])):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for images, labels in validation_loader:
                logits = model(images.to(device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())
        validation_accuracy = accuracy_score(targets, predictions)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader.dataset),
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch + 1}: loss={epoch_record['train_loss']:.4f}, "
            f"val_accuracy={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": int(data_config["image_size"]),
                },
                destination,
            )

    result = {"best_validation_accuracy": best_accuracy, "history": history}
    save_json(result, Path(output_path).with_suffix(".json"))
    return result


def _extract_features(
    root: str | Path, model: WFRNet, config: dict, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    data_config = config["data"]
    loader, classes = feature_loader(
        root,
        int(data_config["image_size"]),
        int(data_config["batch_size"]),
        int(data_config["workers"]),
    )
    all_features, all_labels, all_paths = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=f"Extracting {Path(root).name}"):
            _, features = model(images.to(device), return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_paths.extend(paths)
    return (
        np.concatenate(all_features),
        np.concatenate(all_labels),
        all_paths,
        classes,
    )


def detect_poisoning(
    reference_root: str | Path,
    suspect_root: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    key: str,
    config: dict,
) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WFRNet(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])

    reference_features, reference_labels, _, reference_classes = _extract_features(
        reference_root, model, config, device
    )
    suspect_features, suspect_labels, suspect_paths, suspect_classes = _extract_features(
        suspect_root, model, config, device
    )
    if reference_classes != suspect_classes:
        raise ValueError("Reference and suspect datasets must contain the same classes.")

    guard = FeatureGuard(
        key=key,
        input_dim=reference_features.shape[1],
        projection_dim=int(config["features"]["projection_dim"]),
        quantile=float(config["features"]["distance_quantile"]),
    ).fit(reference_features, reference_labels)
    feature_scores, _ = guard.score(suspect_features, suspect_labels)

    watermark_config = config["watermark"]
    defense_config = config["defense"]
    records = []
    for path, label, feature_score in zip(
        suspect_paths, suspect_labels, feature_scores
    ):
        image_path = Path(path)
        identifier = image_path.stem
        observed_class = suspect_classes[int(label)]
        with Image.open(image_path) as image:
            verification = verify(
                image,
                key,
                identifier,
                observed_class,
                int(watermark_config["repetitions"]),
                float(watermark_config["match_threshold"]),
            )
        watermark_risk = 1.0 - verification.match_rate
        combined_risk = (
            float(defense_config["watermark_weight"]) * watermark_risk
            + float(defense_config["feature_weight"]) * float(feature_score)
        )
        records.append(
            {
                "sample_id": identifier,
                "path": str(image_path),
                "observed_class": observed_class,
                "watermark_match": verification.match_rate,
                "feature_risk": float(feature_score),
                "combined_risk": combined_risk,
                "flagged": combined_risk
                >= float(defense_config["decision_threshold"]),
            }
        )

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    detections = pd.DataFrame(records)
    detections.to_csv(destination, index=False)

    manifest_path = Path(suspect_root) / "manifest.csv"
    metrics: dict[str, float | int] = {"flagged_samples": int(detections["flagged"].sum())}
    if manifest_path.exists():
        truth = pd.read_csv(manifest_path)[["sample_id", "poisoned"]]
        evaluation = detections.merge(truth, on="sample_id", how="inner")
        y_true = evaluation["poisoned"].astype(bool)
        y_pred = evaluation["flagged"].astype(bool)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    save_json(metrics, destination.with_suffix(".metrics.json"))
    return metrics


def quarantine_flagged(
    detections_csv: str | Path, quarantine_root: str | Path
) -> int:
    detections = pd.read_csv(detections_csv)
    flagged = detections[detections["flagged"].astype(bool)]
    destination = Path(quarantine_root)
    count = 0
    for _, row in flagged.iterrows():
        source = Path(str(row["path"]))
        target = destination / str(row["observed_class"]) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        count += 1
    return count

