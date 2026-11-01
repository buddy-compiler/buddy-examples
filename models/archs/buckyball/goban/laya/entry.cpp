#include "linux.h"
#include "model.h"
#include <cstdlib>

extern "C" void laya_run(laya::Output *output, laya::Parameters *parameters,
                         laya::Bias *bias, laya::Weights *weights,
                         laya::Tokens *tokens, laya::Tokens *mask,
                         laya::Tokens *positions, laya::Valid *valid,
                         laya::Type *type) {
  laya::Arguments arguments{output, parameters, bias,  weights, tokens,
                            mask,   positions,  valid, type};
  if (runtime_linux_job(
          0, DEVICE_CORES_0_SIGNATURE,
          [](void *, void *opaque) {
            auto &a = *static_cast<laya::Arguments *>(opaque);
            _mlir_ciface_forward(a.output, a.parameters, a.bias, a.weights,
                                 a.tokens, a.mask, a.positions, a.valid,
                                 a.type);
            return 0;
          },
          &arguments))
    std::abort();
}
