/*
 * src/baseline.cc
 * Maximal accuracy/cost baseline. Full inference every sample.
 *
 * Created: 2026-03-10
 * Authors: Maxence Morel Dierckx, Claude Opus 4.6
 */

#include "pipeline.h"

int main() {
    Pipeline p = pipeline_init();

    constexpr int kInputSize = WINDOW_SIZE * INPUT_CHANNELS;
    constexpr int kOutputSize = WINDOW_SIZE * OUTPUT_CHANNELS;

    float window[kInputSize] = {0};
    float sample[INPUT_CHANNELS];

    while (read_sample(sample, INPUT_CHANNELS)) {
        normalize(sample, INPUT_MEANS, INPUT_STDS, INPUT_CHANNELS);

        // Shift window left, insert new sample at end
        memmove(window, window + INPUT_CHANNELS,
                sizeof(float) * (kInputSize - INPUT_CHANNELS));
        memcpy(window + kInputSize - INPUT_CHANNELS, sample,
               sizeof(float) * INPUT_CHANNELS);

        // Copy window into model input and invoke
        memcpy(p.input, window, sizeof(float) * kInputSize);
        if (pipeline_invoke(&p)) {
            fprintf(stderr, "error: inference failed\n");
            return 1;
        }

        // Last row of output
        float result[OUTPUT_CHANNELS];
        memcpy(result, p.output + kOutputSize - OUTPUT_CHANNELS,
               sizeof(float) * OUTPUT_CHANNELS);
        denormalize(result, OUTPUT_MEANS, OUTPUT_STDS, OUTPUT_CHANNELS);

        write_output(result, OUTPUT_CHANNELS);
    }

    return 0;
}
