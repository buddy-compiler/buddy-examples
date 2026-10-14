import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch._inductor.decomposition import decompositions

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

output = args.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
from importlib import import_module

model, config = import_module("stack.models.models.gemma4-e2b-it.prepare").load_model(
    "google/gemma-4-E2B-it"
)
model.config.use_cache = False
model.config.return_dict = False
model.eval()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
tokenizer.save_pretrained(output / "tokenizer")
with (output / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
for phase, length in (("prefill", 32), ("decode", 1)):
    importer = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        trace=TraceConfig(
            load_trace_config(Path(__file__).parent / "trace/trace.toml")
        ),
    )
    with torch.no_grad():
        graphs = importer.importer(
            model, input_ids=torch.zeros((1, length), dtype=torch.int64)
        )
    assert len(graphs) == 1
    graph = graphs[0]
    params = importer.imported_params[graph]
    graph._func_name = f"forward_{phase}"
    graph.fuse_ops([simply_fuse])
    graph.op_groups[f"subgraph0_{phase}"] = graph.op_groups.pop("subgraph0")
    graph.group_map_device[f"subgraph0_{phase}"] = graph.group_map_device.pop(
        "subgraph0"
    )
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    (output / f"subgraph0_{phase}.mlir").write_text(
        str(driver.subgraphs[0]._imported_module)
    )
    (output / f"forward_{phase}.mlir").write_text(
        str(driver.construct_main_graph(True))
    )
    parameters = np.concatenate(
        [parameter.detach().cpu().numpy().reshape(-1) for parameter in params]
    )
    parameters.astype(np.float32).tofile(output / f"arg0_{phase}.data")

parameters.astype(np.float32).tofile(output / "arg0.data")
