# Beyond Scalar Rewards: Taming Advantage Oscillation in Flow Matching Models via Comparative Policy Optimization

This repository contains our implementation of POISE for image generation.

## What This Project Does

- Trains Flow Matching models with Comparative Policy Optimization.
- Uses ImageReward as the only reward function.
- Supports the training pipeline, dataset handling, and evaluation code needed for our experiments.


## Data

The code expects dataset files in the `dataset/` directory. The exact format is defined by the training configuration in `config/`.

## Training

The main training entry point for this project is the CPO script under `scripts/`.

Example:

```bash
python scripts/train_sd3_cpo.py --config=config/cpo.py
```

## Code Layout

- `poise/`: reward modules, training utilities, and model helpers
- `config/`: training configurations
- `dataset/`: dataset files and preprocessing scripts
- `scripts/`: training entry points

