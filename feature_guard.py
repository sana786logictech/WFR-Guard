from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _seed_from_key(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class ClassProfile:
    centroid: np.ndarray
    threshold: float


class FeatureGuard:
    """Keyed random projection followed by class-conditional distance checks."""

    def __init__(
        self, key: str, input_dim: int, projection_dim: int = 128, quantile: float = 0.95
    ) -> None:
        rng = np.random.default_rng(_seed_from_key(key))
        projection = rng.normal(
            0.0, 1.0 / np.sqrt(projection_dim), size=(input_dim, projection_dim)
        )
        self.projection = projection.astype(np.float32)
        self.quantile = quantile
        self.profiles: dict[int, ClassProfile] = {}

    def transform(self, features: np.ndarray) -> np.ndarray:
        projected = features @ self.projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, 1e-12)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FeatureGuard":
        transformed = self.transform(features)
        for label in np.unique(labels):
            class_features = transformed[labels == label]
            centroid = class_features.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            distances = np.linalg.norm(class_features - centroid, axis=1)
            threshold = float(np.quantile(distances, self.quantile))
            self.profiles[int(label)] = ClassProfile(
                centroid=centroid, threshold=max(threshold, 1e-6)
            )
        return self

    def score(self, features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.transform(features)
        scores = np.ones(len(labels), dtype=np.float32)
        flagged = np.ones(len(labels), dtype=bool)
        for index, (feature, label) in enumerate(zip(transformed, labels)):
            profile = self.profiles.get(int(label))
            if profile is None:
                continue
            distance = float(np.linalg.norm(feature - profile.centroid))
            scores[index] = min(distance / profile.threshold, 2.0) / 2.0
            flagged[index] = distance > profile.threshold
        return scores, flagged

