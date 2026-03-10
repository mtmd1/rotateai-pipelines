# RotateAI Pipelines

Inference pipelines for on-tag whale orientation correction. Designed for STM32U5 deployment, tested with [rotateai-simulator](https://github.com/mtmd1/rotateai-simulator).

Each pipeline reads sensor data from stdin and writes corrected orientation to stdout using binary float32. They differ in when and how often inference runs.

## Installation

```sh
./install.sh
```

This clones and builds [TFLite Micro](https://github.com/tensorflow/tflite-micro). A Python environment is also required for model preparation:

```sh
python -m venv .venv    # requires Python <= 3.13
source .venv/bin/activate
pip install tensorflow numpy
```

## Model Preparation

Converts a Keras model and its preprocessing parameters into C-compatible files for compilation.

```sh
source .venv/bin/activate
python tools/prepare_model.py --model /path/to/model.keras --params /path/to/params.pkl
```

See `python tools/prepare_model.py --help` for all options.

## Build

```sh
make baseline
make variable
make multiple
make surface
```

## Current Pipelines

| Pipeline | Strategy | Description |
| -------- | -------- | ----------- |
| `baseline` | Every sample | Maximum accuracy and cost. |
| `variable` | Every X samples | Measures a sample window periodically. |
| `multiple` | Batched windows | Accumulates multiple windows and runs them together. |
| `surface`  | Event-triggered | Detects surfacing periods and runs inference on them. |
