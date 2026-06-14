from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np
from PIL import Image


PAYLOAD_BITS = 128


def _payload_bits(key: str, sample_id: str, class_name: str) -> np.ndarray:
    message = f"{sample_id}|{class_name}".encode("utf-8")
    digest = hmac.new(key.encode("utf-8"), message, hashlib.sha256).digest()
    return np.unpackbits(np.frombuffer(digest[: PAYLOAD_BITS // 8], dtype=np.uint8))


def _locations(
    key: str, sample_id: str, total_values: int, count: int
) -> np.ndarray:
    seed_material = hmac.new(
        key.encode("utf-8"), sample_id.encode("utf-8"), hashlib.sha256
    ).digest()
    seed = int.from_bytes(seed_material[:8], "big")
    rng = np.random.default_rng(seed)
    if count > total_values:
        raise ValueError("Image is too small for the requested watermark repetitions.")
    return rng.choice(total_values, size=count, replace=False)


@dataclass(frozen=True)
class Verification:
    match_rate: float
    valid: bool


def embed(
    image: Image.Image,
    key: str,
    sample_id: str,
    class_name: str,
    repetitions: int = 5,
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    flat = array.reshape(-1)
    bits = np.tile(_payload_bits(key, sample_id, class_name), repetitions)
    locations = _locations(key, sample_id, flat.size, bits.size)
    flat[locations] = (flat[locations] & 0xFE) | bits
    return Image.fromarray(array, mode="RGB")


def verify(
    image: Image.Image,
    key: str,
    sample_id: str,
    class_name: str,
    repetitions: int = 5,
    threshold: float = 0.80,
) -> Verification:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    flat = array.reshape(-1)
    expected = np.tile(_payload_bits(key, sample_id, class_name), repetitions)
    locations = _locations(key, sample_id, flat.size, expected.size)
    observed = flat[locations] & 1
    match_rate = float(np.mean(observed == expected))
    return Verification(match_rate=match_rate, valid=match_rate >= threshold)

