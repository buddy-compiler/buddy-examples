# ===- buddy-pytorch-importer.py ---------------------------------------------------
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
# This is the DeepSeekR1 AOT importer.
#
# ===---------------------------------------------------------------------------

from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch._inductor.decomposition import decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
deepseek = import_module("models.models.deepseek-r1-distill-qwen-1-5b.model")

# forward_prefill takes the whole prompt, any number of tokens: the token
# count is a dynamic size (8 tokens only give the trace an example).
# forward_decode takes one token. Both write their tokens from a start
# position into the KV cache.
for name, length in (("prefill", 8), ("decode", 1)):
    # ===------------------------------------------------------------------===
    # Step 1: Import the decoder.
    # ===------------------------------------------------------------------===
    model = deepseek.load()
    compiler = DynamoCompiler(
        primary_registry={**tosa.ops_registry, **quant.ops_registry},
        aot_autograd_decomposition=decompositions,
        trace=None,
        func_name=f"forward_{name}",
    )
    tokens, start, keys, values = deepseek.inputs(length)
    if name == "prefill":
        torch._dynamo.mark_dynamic(tokens, 1, min=1, max=deepseek.capacity)
    with torch.no_grad():
        (graph,) = compiler.importer(model, tokens, start, keys, values)
    params = compiler.imported_params[graph]

    # ===------------------------------------------------------------------===
    # Step 2.1: Export the original graph.
    # ===------------------------------------------------------------------===
    graph.dump(output / f"01-import-{name}.txt")

    # ===------------------------------------------------------------------===
    # Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
    # ===------------------------------------------------------------------===
    permute(graph)
    graph.dump(output / f"02-permuted-{name}.txt")

    # ===------------------------------------------------------------------===
    # Step 2.3: Quantize every matrix product to MXINT8, blocked along K.
    # The perplexity gate is buddy-pytorch-calibrate.py.
    # ===------------------------------------------------------------------===
    params = quantize(graph, params)
    graph.dump(output / f"03-quanted-{name}.txt")

    # ===------------------------------------------------------------------===
    # Step 3: Lower the graph; both functions take the same parameters.
    # ===------------------------------------------------------------------===
    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    (output / f"subgraph0_{name}.mlir").write_text(
        str(driver.subgraphs[0]._imported_module)
    )
    (output / f"forward_{name}.mlir").write_text(str(driver.construct_main_graph(True)))
# Both functions take the parameters grouped by type: FP32, INT8 weights.
for index, dtype in enumerate((torch.float32, torch.int8)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")
