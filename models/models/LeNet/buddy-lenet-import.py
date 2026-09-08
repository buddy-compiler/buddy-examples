# ===- buddy-lenet-import.py ---------------------------------------------------
#
# This is the LeNet model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from framework.quant.core.importer import quantize_model_graph
from framework.quant.core.activation import calibrate_layers
from model import LeNet

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
args = parser.parse_args()

# Retrieve the LeNet model path from environment variables.
model_path = os.environ.get("LENET_MODEL_PATH")
if model_path is None:
    raise EnvironmentError(
        "The environment variable 'LENET_MODEL_PATH' is not set or is invalid."
    )
output_dir = Path(model_path)
source_dir = Path(__file__).resolve().parent

model = LeNet()

model = torch.load(output_dir / "lenet-model.pth", weights_only=False)
model = model.eval()

if args.trace:
    trace = TraceConfig(load_trace_config(source_dir / "trace" / args.trace_config))
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

rgb = np.asarray(Image.open(source_dir / "images" / "8.bmp").convert("RGB"), dtype=np.float32)
gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0
data = torch.from_numpy((gray * 2.0 - 1.0)[None, None, :, :].astype(np.float32))
calibration = calibrate_layers(model, data)
# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)


assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
quantize_model_graph(
    graph,
    params,
    [name for name, _ in model.named_parameters()],
    output_dir,
    "lenet",
    calibration,
)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
with open(output_dir / "subgraph0.mlir", "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)

with open(output_dir / "forward.mlir", "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)
