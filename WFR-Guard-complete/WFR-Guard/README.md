# WFR-Guard

WFR-Guard is a dual-domain defense framework for detecting and mitigating
data-poisoning attacks in image classification systems. It combines keyed LSB
watermark verification with secret randomized deep-feature projection,
class-conditional anomaly scoring, and suspicious-sample quarantine.

## Main Components

- **Watermark protection:** Binds each image identity to its original class.
- **Attack simulation:** Supports label flipping, Gaussian noise, and patch triggers.
- **Feature randomization:** Applies a keyed random projection to CNN embeddings.
- **Dual verification:** Combines watermark mismatch and feature anomaly risk.
- **Mitigation:** Produces detection reports and quarantines flagged samples.
- **Evaluation:** Reports accuracy, precision, recall, and F1 when attack labels exist.

## Repository Structure

```text
WFR-Guard/
├── configs/default.yaml
├── scripts/create_demo_dataset.py
├── tests/
├── wfr_guard/
│   ├── cli.py
│   ├── data.py
│   ├── feature_guard.py
│   ├── model.py
│   ├── pipeline.py
│   ├── utils.py
│   └── watermark.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Dataset Format

Arrange the clean dataset as an ImageFolder dataset:

```text
data/raw/
├── class_1/
│   ├── image_001.png
│   └── image_002.png
└── class_2/
    ├── image_003.png
    └── image_004.png
```

The repository intentionally excludes research datasets, participant data,
trained weights, and secret keys.

## Quick Demonstration

Create a small synthetic dataset:

```bash
python scripts/create_demo_dataset.py
```

Set a private key:

```bash
export WFR_GUARD_KEY="replace-with-a-long-private-key"
```

Protect the clean images:

```bash
wfr-guard protect \
  --input data/raw \
  --output data/protected
```

Create a controlled label-flipping attack:

```bash
wfr-guard attack \
  --input data/protected \
  --output data/poisoned \
  --type label_flip \
  --ratio 0.20
```

Train the classifier on the trusted protected set:

```bash
wfr-guard train \
  --data data/protected \
  --output outputs/wfr_model.pt
```

Run dual-domain poisoning detection:

```bash
wfr-guard detect \
  --reference data/protected \
  --suspect data/poisoned \
  --model outputs/wfr_model.pt \
  --output outputs/detections.csv
```

Copy flagged samples into quarantine:

```bash
wfr-guard quarantine \
  --detections outputs/detections.csv \
  --output outputs/quarantine
```

## Attack Options

```text
label_flip
gaussian_noise
patch_trigger
```

For a targeted label-flipping experiment:

```bash
wfr-guard attack \
  --input data/protected \
  --output data/poisoned_targeted \
  --type label_flip \
  --ratio 0.20 \
  --source-class class_1 \
  --target-class class_2
```

## Outputs

- `manifest.csv`: sample identity, true label, observed label, and attack status.
- `detections.csv`: watermark, feature, and combined risk for each sample.
- `detections.metrics.json`: detection accuracy, precision, recall, and F1.
- `wfr_model.pt`: trained model checkpoint.
- `wfr_model.json`: training history and best validation accuracy.

## Tests

```bash
pytest -q
```

## Security Notes

- Never commit the watermark key, API credentials, or private datasets.
- Use lossless PNG files after LSB watermark insertion.
- Keep a trusted clean reference subset for fitting feature profiles.
- Replace the demonstration classifier or thresholds as needed for the target dataset.

## Citation

If this code supports your research, cite the associated WFR-Guard paper. A
BibTeX entry can be added after publication details become available.
