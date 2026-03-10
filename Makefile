TFLM_DIR   = deps/tflite-micro
TFLM_GEN   = $(TFLM_DIR)/gen/linux_x86_64_default_gcc
TFLM_LIB   = $(TFLM_GEN)/lib/libtensorflow-microlite.a
TFLM_DL    = $(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads
GEN_DIR    = build/models

CXXFLAGS   = -std=c++17 -DTF_LITE_STATIC_MEMORY \
             -I$(TFLM_DIR) \
             -I$(TFLM_DL)/flatbuffers/include \
             -I$(TFLM_DL)/gemmlowp \
             -I$(TFLM_DL)/kissfft \
             -I$(TFLM_DL)/ruy \
             -I$(GEN_DIR) \
             -Isrc

# Pipelines
baseline: build/baseline
build/baseline: src/baseline.cc src/pipeline.h $(GEN_DIR)/model_data.inc $(TFLM_LIB)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/baseline.cc -o $@ $(TFLM_LIB)

# TFLite Micro static library
$(TFLM_LIB):
	make -C $(TFLM_DIR) -f tensorflow/lite/micro/tools/make/Makefile TARGET=linux microlite

clean:
	rm -rf build/

.PHONY: baseline clean
