#include "ModelImage.h"
#include "checked.h"

alignas(8) static const unsigned char prompt_data[] = {
#embed "check-prompt.data"
};

int main() {
  uint64_t weight_bytes, parameter_bytes;
  const auto *weights =
      model::imageFile(MODEL_BASE, "arg0.data", &weight_bytes);
  const auto *parameters =
      model::imageFile(MODEL_BASE, "arg1.data", &parameter_bytes);
  return check_model(reinterpret_cast<const uint16_t *>(weights),
                     weight_bytes / sizeof(uint16_t),
                     reinterpret_cast<const float *>(parameters),
                     parameter_bytes / sizeof(float),
                     reinterpret_cast<const int64_t *>(prompt_data),
                     sizeof(prompt_data) / sizeof(int64_t));
}
