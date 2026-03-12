# Inference with TFLite Micro

[TFLite Micro](https://github.com/tensorflow/tflite-micro) is TensorFlow's C++ inference engine for microcontrollers. It runs neural networks on devices with no OS, no heap allocator, and as little as 16 KB of RAM. The runtime compiles into a single static library (`libtensorflow-microlite.a`) to be linked against.

TFLite Micro does not support dynamic memory allocation or the filesystem. 
- Tensors during model inference are held in a fixed size "arena" that is passed to the model interface. 
- Model weights are embedded as a C array at compile time. 
- Model operators are manually set and only those used are linked in, potentially reducing binary size by up to 200 KB.

## How Pipelines Use It

### 1. Model Preparation

The models are produced as `.keras`. They must be converted to TFLite format and then to a C byte array. `tools/prepare_model.py` does this in one step:

```sh
python tools/prepare_model.py --model model.keras --params params.pkl
```

Producing two files in `build/models/`:
- `model_data.inc` — the model weights as a comma-separated byte array, included directly into the binary with `#include`
- `model_params.h` — z-score means and stds, model dimensions, and a macro that registers only the operators the model needs

`prepare_model.py` copies TFLite Micro's own [generate_micro_mutable_op_resolver_from_model](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver) tool to determine the ops a model will use programmatically.

### 2. Setup

Inference requires setting up the model interpreter and its I/O pointers:

```cpp
// 1. Loading the model from the embedded byte array
const tflite::Model* model = tflite::GetModel(model_data);

// 2. Registering only required operators
tflite::MicroMutableOpResolver<NUM_OPS> resolver;
REGISTER_OPS(resolver);  // macro from model_params.h

// 3. Creating an interpreter with a static memory arena
constexpr int kArenaSize = 256 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter interp(model, resolver, tensor_arena, kArenaSize);
interp.AllocateTensors();

// 4. Getting pointers to the input/output tensors
float* input = interp.input(0)->data.f;
float* output = interp.output(0)->data.f;
```

### 3. Inference

The interface for models is extremely simple. Copy data into the input tensor, invoke, then read from the output tensor.

```cpp
memcpy(input, my_data, sizeof(float) * input_size);
interp.Invoke();
// output now contains the model's prediction
```

`Invoke()` runs the forward pass synchronously and returns 0 on success, 1 on failure (as an enum `ktfLiteOk`/`ktfLiteError`).

### 4. Our Implementation

`prepare_model.py` does the model preparation. `src/pipeline.h` performs the model setup in `pipeline_init()`, which returns a struct with the interpreter and tensor pointers. Each pipeline includes `pipeline.h` and writes its own main loop.
