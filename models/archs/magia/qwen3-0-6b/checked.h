#pragma once

#include <stdint.h>

int check_model(const uint16_t *weights, intptr_t weight_count,
                const float *parameters, intptr_t parameter_count,
                const int64_t *input, unsigned input_count);
