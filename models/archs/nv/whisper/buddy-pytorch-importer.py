# ===- buddy-pytorch-importer.py -------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#

from importlib import import_module
from pathlib import Path
import json

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import AutoTokenizer
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
whisper = import_module("models.models.whisper.model")
model = whisper.load()
# The attention runs through explicit products the quantization can read as
# INT8, not through one fused attention.
model.model.config._attn_implementation = "eager"
dynamo_compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=inductor_decomp,
)
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, *whisper.inputs())
if len(graphs) != 1:
    raise RuntimeError(f"Expected one graph, got {len(graphs)}")
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.dump(output_dir / "01-import.txt")
# Absorb the transposes PyTorch puts on the Linear weights.
permute(graph)
graph.dump(output_dir / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every matrix product: the Linears and the attention
# products to MXINT8, the convolutions to INT8.
# ===----------------------------------------------------------------------===
# The activation scales are computed online; buddy-pytorch-calibrate.py
# accepts the quantized model first.
calibration = json.loads((output_dir / "calibration.json").read_text())
print(f"Quantized transcript accepted: {calibration['transcript']}")
params = quantize(graph, params)
graph.dump(output_dir / "03-quanted.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
with open(output_dir / "subgraph0.mlir", "w") as f:
    print(driver.subgraphs[0]._imported_module, file=f)
with open(output_dir / "forward.mlir", "w") as f:
    print(driver.construct_main_graph(True), file=f)
# forward takes the parameters grouped by type: FP32 (the biases among
# them), INT8 weights (MXINT8 codes and exponents for the Linears).
for index, dtype in enumerate((torch.float32, torch.int8)):
    numpy.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output_dir / f"arg{index}.data")
print(f"[import-whisper] Wrote forward.mlir, subgraph0.mlir, arg*.data → {output_dir}")

tokenizer = AutoTokenizer.from_pretrained(whisper.checkpoint)
with (output_dir / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
