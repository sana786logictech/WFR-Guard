from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def image_transform(image_size: int, training: bool = False):
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


def training_loaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    validation_split: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    full_dataset = datasets.ImageFolder(
        root=str(root), transform=image_transform(image_size, training=False)
    )
    validation_size = max(1, int(len(full_dataset) * validation_split))
    training_size = len(full_dataset) - validation_size
    if training_size < 1:
        raise ValueError("The dataset needs at least two images.")

    generator = torch.Generator().manual_seed(seed)
    training_set, validation_set = random_split(
        full_dataset, [training_size, validation_size], generator=generator
    )
    train_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return train_loader, validation_loader, full_dataset.classes


def feature_loader(
    root: str | Path, image_size: int, batch_size: int, workers: int
) -> tuple[DataLoader, list[str]]:
    dataset: Dataset = ImageFolderWithPaths(
        root=str(root), transform=image_transform(image_size, training=False)
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return loader, dataset.classes  # type: ignore[attr-defined]
