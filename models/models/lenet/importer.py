from importlib import import_module


# ===- buddy-lenet-import.py ---------------------------------------------------
#
# This is the LeNet model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path
import sys

import torch


from .model import LeNet

import argparse

parser = argparse.ArgumentParser(description="LeNet model AOT importer")
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/trace.toml.",
)
parser.add_argument(
    "--trace-config",
    type=str,
    default="trace.toml",
    help="Trace config file under trace/.",
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--compiler-build", type=Path, required=True)
parser.add_argument("--design", required=True)
args = parser.parse_args()
args.compiler_build = args.compiler_build.resolve()
sys.path.insert(0, str(args.compiler_build.resolve() / "python_packages"))
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

design = import_module(args.design)
quant = import_module(design.__name__ + ".quant.quantize")
design_dir = Path(design.__file__).parent
torch.set_num_threads(4)

output_dir = args.output_dir.resolve()
output_dir.mkdir(parents=True, exist_ok=True)
source_dir = Path(__file__).resolve().parent

model = LeNet()

model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
model = model.eval()

if args.trace:
    trace = TraceConfig(load_trace_config(design_dir / "trace" / args.trace_config))
    verbose = False
    verbose_path = None
else:
    trace = None
    verbose = True
    verbose_path = os.path.join(output_dir, "output", "buddy-graph.txt")
    if os.path.exists(verbose_path):
        os.remove(verbose_path)

dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

data, calibration = quant.prepare(model, source_dir)
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)


assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
quant.quantize(
    graph,
    params,
    [name for name, _ in model.named_parameters()],
    output_dir,
    "lenet",
    calibration,
    rax_pack=args.compiler_build / "bin/rax-pack",
)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

import json
metadata = {"version": 1, "chip": design.chip, "model": "lenet"}
(output_dir / "model.json").write_text(json.dumps(metadata))
