from importlib import import_module
from pathlib import Path

import numpy as np

import torch
from torch._inductor.decomposition import decompositions
from transformers import BertForSequenceClassification, BertTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

BertLogits = import_module("models.models.bert-base-uncased-emotion.model").BertLogits

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)
checkpoint = (
    model_dir.parents[3] / "models/bert-base-uncased-emotion/output"
    "/bert-base-uncased-emotion"
)
sequence_length = 32

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
model = BertLogits(
    BertForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    )
).eval()
inputs = (
    torch.zeros(1, sequence_length, dtype=torch.int64),
    torch.zeros(1, sequence_length, dtype=torch.int64),
    torch.ones(1, sequence_length, dtype=torch.int64),
    torch.arange(sequence_length, dtype=torch.int64)[None],
)
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *inputs)
# forward takes the inputs in the order of BertLogits.forward.
graph.order_inputs(inputs)
params = compiler.imported_params[graph]

# ===----------------------------------------------------------------------===
# Step 2: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
# No goban Ball is recognized yet, so the whole graph stays FP32 and runs in
# software: no layout alignment and no quantization.
graph.dump(output / "01-import.txt")
# graph.dump_parameters(output / "01-import-parameters")

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
np.concatenate([p.detach().numpy().reshape(-1) for p in params]).astype(
    np.float32, casting="no"
).tofile(output / "arg0.data")

# The runner tokenizes the prompt with the vocabulary of the checkpoint.
vocabulary = BertTokenizer.from_pretrained(checkpoint).get_vocab()
(output / "vocab.txt").write_text(
    "".join(token + "\n" for token in sorted(vocabulary, key=vocabulary.get))
)
