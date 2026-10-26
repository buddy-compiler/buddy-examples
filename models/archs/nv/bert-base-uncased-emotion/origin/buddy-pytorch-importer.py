from pathlib import Path

import numpy as np
import torch


from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

output = Path("output").resolve()
output.mkdir(parents=True, exist_ok=True)
from transformers import BertForSequenceClassification
from torch._inductor.decomposition import decompositions


from importlib import import_module

BertLogits = import_module("models.models.bert-base-uncased-emotion.model").BertLogits
checkpoint = (
    Path(__file__).resolve().parent
    / "../../../../models/bert-base-uncased-emotion/output/bert-base-uncased-emotion"
).resolve()
model = BertLogits(
    BertForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    )
)
model.eval()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(Path(__file__).parent / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(
        model,
        input_ids=torch.zeros((1, 32), dtype=torch.int64),
        token_type_ids=torch.zeros((1, 32), dtype=torch.int64),
        attention_mask=torch.ones((1, 32), dtype=torch.int64),
        position_ids=torch.arange(32, dtype=torch.int64)[None, :],
    )
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)

for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))

parameters = np.concatenate(
    [parameter.detach().cpu().numpy().reshape(-1) for parameter in params]
)
parameters.astype(np.float32).tofile(output / "arg0.data")
