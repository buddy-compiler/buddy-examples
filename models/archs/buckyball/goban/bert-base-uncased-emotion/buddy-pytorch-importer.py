from importlib import import_module
from pathlib import Path
import json

import numpy as np

import torch
from torch._inductor.decomposition import decompositions
from transformers import BertTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
bert = import_module("models.models.bert-base-uncased-emotion.model")
model = bert.load()
inputs = bert.inputs()
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *inputs)
# forward takes the inputs in the order of BertLogits.forward.
graph.order_inputs(inputs)
params = compiler.imported_params[graph]
names = {parameter.data_ptr(): name for name, parameter in model.named_parameters()}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every Linear and attention product to INT8: the goban
# matmul thread computes in INT8 only.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
calibration = json.loads((output / "calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one goban core runs all of it.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
graph.dump(output / "04-partitioned.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# forward takes the parameters grouped by type: FP32, INT32 biases, INT8
# weights.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")

# The runner tokenizes the prompt with the vocabulary of the checkpoint.
tokenizer = BertTokenizer.from_pretrained(bert.checkpoint)
vocabulary = tokenizer.get_vocab()
(output / "vocab.txt").write_text(
    "".join(token + "\n" for token in sorted(vocabulary, key=vocabulary.get))
)
sample = tokenizer(
    (bert.checkpoint.parent.parent / "input.txt").read_text().strip(),
    padding="max_length",
    max_length=32,
    truncation=True,
)
np.asarray(sample["input_ids"], dtype=np.int64).tofile(output / "input_ids.data")
np.asarray(sample["attention_mask"], dtype=np.int64).tofile(
    output / "attention_mask.data"
)
