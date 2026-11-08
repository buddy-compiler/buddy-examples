from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch._inductor.decomposition import decompositions
from transformers import AutoTokenizer

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
gemma4 = import_module("models.models.gemma4-e2b-it.model")

# The prompt runs in chunks of this many tokens (docs/ChunkedPrefill.md of
# Buddy-MLIR): forward_prefill takes a chunk, forward_decode one token. Both
# write their tokens from a start position into the KV cache.
chunk = 64

for name, length in (("prefill", chunk), ("decode", 1)):
    # ===------------------------------------------------------------------===
    # Step 1: Import the decoder for `length` tokens.
    # ===------------------------------------------------------------------===
    model = gemma4.load()
    compiler = DynamoCompiler(
        primary_registry={**tosa.ops_registry, **quant.ops_registry},
        aot_autograd_decomposition=decompositions,
        trace=None,
        func_name=f"forward_{name}",
    )
    with torch.no_grad():
        (graph,) = compiler.importer(model, *gemma4.inputs(length))
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
tokenizer = AutoTokenizer.from_pretrained(gemma4.checkpoint)
with (output / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
