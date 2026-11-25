#include "checked.h"

// The weights of the Linears, which the mesh reads in place.
alignas(64) static const unsigned char weight_data[]
    __attribute__((section(".parameters"))) = {
#embed "arg0.data"
};
alignas(64) static const unsigned char parameter_data[] = {
#embed "arg1.data"
};
alignas(8) static const unsigned char id_data[] = {
#embed "check-ids.data"
};

int main() {
  return check_model(reinterpret_cast<const uint16_t *>(weight_data),
                     sizeof(weight_data) / sizeof(uint16_t),
                     reinterpret_cast<const float *>(parameter_data),
                     sizeof(parameter_data) / sizeof(float),
                     reinterpret_cast<const int64_t *>(id_data),
                     sizeof(id_data) / sizeof(int64_t));
}
