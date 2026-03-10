#!/bin/bash
set -e

# TFLite Micro
git clone --depth 1 https://github.com/tensorflow/tflite-micro.git deps/tflite-micro
make -C deps/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile TARGET=linux microlite

# Python environment for tools/
python3 -m venv .venv
.venv/bin/pip install tensorflow numpy
