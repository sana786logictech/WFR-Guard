from __future__ import annotations

import argparse
import json

from .pipeline import (
    detect_poisoning,
    poison_dataset,
    protect_dataset,
    quarantine_flagged,
    train_model,
)
from .utils import get_secret_key, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wfr-guard",
        description="Watermark and feature-randomization defense against data poisoning.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    protect = commands.add_parser("protect", help="Embed keyed watermarks.")
    protect.add_argument("--input", required=True)
    protect.add_argument("--output", required=True)
    protect.add_argument("--key")

    attack = commands.add_parser("attack", help="Create a controlled poisoned dataset.")
    attack.add_argument("--input", required=True)
    attack.add_argument("--output", required=True)
    attack.add_argument(
        "--type", choices=["label_flip", "gaussian_noise", "patch_trigger"], required=True
    )
    attack.add_argument("--ratio", type=float, default=0.10)
    attack.add_argument("--source-class")
    attack.add_argument("--target-class")

    train = commands.add_parser("train", help="Train the image classifier.")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)

    detect = commands.add_parser("detect", help="Run dual-domain verification.")
    detect.add_argument("--reference", required=True)
    detect.add_argument("--suspect", required=True)
    detect.add_argument("--model", required=True)
    detect.add_argument("--output", required=True)
    detect.add_argument("--key")

    quarantine = commands.add_parser("quarantine", help="Copy flagged samples.")
    quarantine.add_argument("--detections", required=True)
    quarantine.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "protect":
        key = get_secret_key(args.key, config["watermark"]["key_env"])
        result = protect_dataset(
            args.input, args.output, key, int(config["watermark"]["repetitions"])
        )
        print(f"Manifest: {result}")
    elif args.command == "attack":
        if not 0 < args.ratio <= 1:
            raise ValueError("--ratio must be greater than 0 and no more than 1.")
        result = poison_dataset(
            args.input,
            args.output,
            args.type,
            args.ratio,
            int(config["seed"]),
            args.source_class,
            args.target_class,
        )
        print(f"Manifest: {result}")
    elif args.command == "train":
        result = train_model(args.data, args.output, config)
        print(json.dumps(result, indent=2))
    elif args.command == "detect":
        key = get_secret_key(args.key, config["watermark"]["key_env"])
        result = detect_poisoning(
            args.reference,
            args.suspect,
            args.model,
            args.output,
            key,
            config,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "quarantine":
        count = quarantine_flagged(args.detections, args.output)
        print(f"Quarantined {count} samples.")


if __name__ == "__main__":
    main()

