#include "linux.h"
#include "model.h"

#include <cstdlib>

extern "C" void bert_run(bert::Output *output, bert::Parameters *parameters,
                         bert::Bias *bias, bert::Weights *weights,
                         bert::Tokens *ids, bert::Tokens *types,
                         bert::Tokens *mask, bert::Tokens *positions) {
  bert::Arguments arguments{output, parameters, bias, weights,
                            ids,    types,      mask, positions};
  const int status = runtime_linux_job(
      0, DEVICE_CORES_0_SIGNATURE,
      [](void *, void *opaque) {
        auto &args = *static_cast<bert::Arguments *>(opaque);
        _mlir_ciface_forward(args.output, args.parameters, args.bias,
                             args.weights, args.ids, args.types, args.mask,
                             args.positions);
        return 0;
      },
      &arguments);
  if (status)
    std::abort();
}
