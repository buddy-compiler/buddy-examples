#include "linux.h"
#include "model.h"

#include <cstdlib>

// The core runtime supplies the engine used by the compiled thread streams.
extern "C" void mixer_run(mixer::Output *output, mixer::Parameters *parameters,
                          mixer::Bias *bias, mixer::Weights *weights,
                          mixer::Input *input) {
  mixer::Arguments arguments{output, parameters, bias, weights, input};
  const int status = runtime_linux_job(
      0, DEVICE_CORES_0_SIGNATURE,
      [](void *, void *opaque) {
        auto &args = *static_cast<mixer::Arguments *>(opaque);
        _mlir_ciface_forward(args.output, args.parameters, args.bias,
                             args.weights, args.input);
        return 0;
      },
      &arguments);
  if (status)
    std::abort();
}
