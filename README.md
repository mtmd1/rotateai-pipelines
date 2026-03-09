# RotateAI Pipelines

Inference pipelines for on-tag whale orientation correction. Designed for STM32U5 deployment, tested with [rotateai-simulator](https://github.com/mtmd1/rotateai-simulator).

Each pipeline reads sensor data from stdin and writes corrected orientation to stdout using binary float32. They differ in when and how often inference runs.

## Build

Requires [TFLite Micro](https://github.com/tensorflow/tflite-micro) built for your target.

```sh
make baseline
make variable
make multi
make surface
```

## Current Pipelines

| Pipeline | Strategy | Description |
| -------- | -------- | ----------- |
| `baseline` | Every sample | Maximum accuracy and cost. |
| `variable` | Every X samples | Measures a sample window periodically. |
| `multi`    | Batched windows | Accumulates multiple windows and runs them together. Reduces duty cycle. |
| `surface`  | Event-triggered | Detects surfacing periods and runs inference on them. |

Some pipelines, like `surface`, include other pipelines as modules.

