import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo_root.resolve()))
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

torch.set_num_threads(4)
output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
from transformers import BertForSequenceClassification
from torch._inductor.decomposition import decompositions


class BertLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bhadresh-savani/bert-base-uncased-emotion", attn_implementation="eager"
        )

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids):
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits


model = BertLogits()
data = {
    "input_ids": torch.zeros((1, 32), dtype=torch.int64),
    "token_type_ids": torch.zeros((1, 32), dtype=torch.int64),
    "attention_mask": torch.ones((1, 32), dtype=torch.int64),
    "position_ids": torch.arange(32, dtype=torch.int64)[None, :],
}
model.eval()
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(Path(__file__).parent / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(model, **data)
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
payload = output / "bert.payload"
payload.mkdir(exist_ok=True)
parameters.astype(np.float32).tofile(payload / "params.f32")
(payload / "weights.bin").write_bytes(b"")
(output / "model.json").write_text(
    json.dumps({"version": 1, "chip": "gemmini", "model": "bert"})
)

(output / "bert-parameters.h").write_text(
    "#pragma once\n"
    f"#define BERT_F32_ELEMENTS {parameters.size}\n"
    "#define BERT_SEQUENCE_LENGTH 32\n"
    "#define BERT_NUM_LABELS 6\n"
)
