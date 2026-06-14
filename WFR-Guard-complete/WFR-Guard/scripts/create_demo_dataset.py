from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--samples-per-class", type=int, default=30)
    args = parser.parse_args()

    output = Path(args.output)
    rng = np.random.default_rng(42)
    classes = {
        "circles": (40, 140, 220),
        "squares": (220, 80, 70),
    }
    for class_name, color in classes.items():
        class_dir = output / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.samples_per_class):
            background = rng.integers(225, 256, size=(128, 128, 3), dtype=np.uint8)
            image = Image.fromarray(background)
            draw = ImageDraw.Draw(image)
            offset = int(rng.integers(-8, 9))
            box = (32 + offset, 32, 96 + offset, 96)
            if class_name == "circles":
                draw.ellipse(box, fill=color)
            else:
                draw.rectangle(box, fill=color)
            image.save(class_dir / f"{class_name}_{index:03d}.png")
    print(f"Created demonstration dataset at {output}")


if __name__ == "__main__":
    main()

