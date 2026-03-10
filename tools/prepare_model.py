'''
tools/prepare_model.py
Converts a model.keras and preprocessing_params.pkl
into model_data.inc and model_params.h for use by pipelines.

inputs:
    --model   path to .keras file
    --params  path to .pkl file
    --out     output directory (default: build/models/)

steps:
    1. load .keras model
    2. read input shape, output shape from model
    3. convert to .tflite via TFLiteConverter
    4. write .tflite bytes as C hex array → model_data.inc
    5. load .pkl, extract means and stds
    6. split means/stds into input (7) and output (6) sets
    7. write model_params.h:
        WINDOW_SIZE, INPUT_CHANNELS, OUTPUT_CHANNELS
        INPUT_MEANS[],  INPUT_STDS[]
        OUTPUT_MEANS[], OUTPUT_STDS[]

outputs:
    build/models/model_data.inc
    build/models/model_params.h

Created: 2026-03-10
Authors: Maxence Morel Dierckx, Claude Opus 4.6
'''
import argparse
import os
import pickle
import re
import sys

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.lite.tools import visualize


def parse_op_name(word):
    """Convert flatbuffer op name to resolver method name.

    e.g. CONV_2D → AddConv2D, MAX_POOL_2D → AddMaxPool2D.
    Lifted from tflite-micro generate_micro_mutable_op_resolver_from_model.py.
    """
    word = word.replace('TFLite', '')
    parts = re.split('_|-', word)
    result = ''
    for part in parts:
        if len(part) > 1:
            if part[0].isalpha():
                result += part[0].upper() + part[1:].lower()
            else:
                result += part.upper()
        else:
            result += part.upper()
    result = result.replace('Lstm', 'LSTM')
    result = result.replace('BatchMatmul', 'BatchMatMul')
    return 'Add' + result


def extract_ops(tflite_bytes):
    """Extract operator names from a .tflite flatbuffer.

    Returns sorted list of (op_name, method_name) tuples.
    Warns on unknown/custom ops.
    """
    data = visualize.CreateDictFromFlatbuffer(bytearray(tflite_bytes))
    ops = set()
    for op_code in data['operator_codes']:
        if op_code['custom_code'] is not None:
            name = visualize.NameListToString(op_code['custom_code'])
            print(f'warning: custom op "{name}", skipping', file=sys.stderr)
            continue
        code = max(op_code['builtin_code'], op_code['deprecated_builtin_code'])
        ops.add(visualize.BuiltinCodeToName(code))
    return sorted((op, parse_op_name(op)) for op in ops)


def load_params(path):
    with open(path, 'rb') as f:
        params = pickle.load(f)
    means = np.array(params['means']).flatten()
    stds = np.array(params['stds']).flatten()
    means_label = np.array(params['means_label']).flatten()
    stds_label = np.array(params['stds_label']).flatten()
    window_size = int(params['window_size'])
    return means, stds, means_label, stds_label, window_size


def convert_to_tflite(model, input_channels, window_size):
    fixed_input = tf.TensorSpec([1, window_size, input_channels], tf.float32, name='input')

    @tf.function(input_signature=[fixed_input])
    def inference(x):
        return model(x, training=False)

    concrete_func = inference.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    return converter.convert()


def write_model_data_inc(tflite_bytes, path):
    with open(path, 'w') as f:
        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i+12]
            line = ', '.join(f'0x{b:02x}' for b in chunk)
            if i + 12 < len(tflite_bytes):
                line += ','
            f.write(f'  {line}\n')


def format_float_array(arr):
    return ', '.join(f'{np.float32(v):.8g}f' for v in arr)


def write_model_params_h(path, window_size, input_means, input_stds,
                         output_means, output_stds, ops):
    input_channels = len(input_means)
    output_channels = len(output_means)
    with open(path, 'w') as f:
        f.write('#ifndef MODEL_PARAMS_H\n')
        f.write('#define MODEL_PARAMS_H\n\n')
        f.write(f'#define WINDOW_SIZE {window_size}\n')
        f.write(f'#define INPUT_CHANNELS {input_channels}\n')
        f.write(f'#define OUTPUT_CHANNELS {output_channels}\n')
        f.write(f'#define NUM_OPS {len(ops)}\n\n')
        # REGISTER_OPS macro
        f.write('#define REGISTER_OPS(resolver) \\\n')
        for i, (_, method) in enumerate(ops):
            slash = ' \\' if i < len(ops) - 1 else ''
            f.write(f'    resolver.{method}();{slash}\n')
        f.write('\n')
        f.write(f'static const float INPUT_MEANS[] = {{{format_float_array(input_means)}}};\n')
        f.write(f'static const float INPUT_STDS[] = {{{format_float_array(input_stds)}}};\n')
        f.write(f'static const float OUTPUT_MEANS[] = {{{format_float_array(output_means)}}};\n')
        f.write(f'static const float OUTPUT_STDS[] = {{{format_float_array(output_stds)}}};\n\n')
        f.write('#endif\n')


def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to C-compatible files')
    parser.add_argument('--model', required=True, help='Path to .keras file')
    parser.add_argument('--params', required=True, help='Path to .pkl file')
    parser.add_argument('--out', default='build/models/', help='Output directory')
    args = parser.parse_args()

    try:
        # Step 1: Load preprocessing params
        means, stds, means_label, stds_label, window_size = load_params(args.params)
        input_channels = len(means)
        print(f'Loaded params: window_size={window_size}, '
              f'input_channels={input_channels}, output_channels={len(means_label)}')

        # Step 2: Load and convert model
        model = tf.keras.models.load_model(args.model)
        print(f'Loaded model from {args.model}')

        tflite_bytes = convert_to_tflite(model, input_channels, window_size)
        print(f'Converted to TFLite ({len(tflite_bytes)} bytes)')

        # Step 3: Extract ops from .tflite
        ops = extract_ops(tflite_bytes)
        print(f'Extracted {len(ops)} ops: {", ".join(m for _, m in ops)}')

        # Step 4: Write outputs
        os.makedirs(args.out, exist_ok=True)

        inc_path = os.path.join(args.out, 'model_data.inc')
        write_model_data_inc(tflite_bytes, inc_path)
        print(f'Wrote {inc_path}')

        params_path = os.path.join(args.out, 'model_params.h')
        write_model_params_h(params_path, window_size, means, stds,
                             means_label, stds_label, ops)
        print(f'Wrote {params_path}')

    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
