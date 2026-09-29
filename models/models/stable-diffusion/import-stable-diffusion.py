# ===- import-stable-diffusion.py ----------------------------------------------
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
# This is the Stable Diffusion model AOT importer.
#
# ===---------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
import numpy
import torch
import torchvision.models as models
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.graph.type import DeviceType
from buddy.compiler.ops import tosa
from buddy.compiler.graph.operation import *
from buddy.compiler.trace import TraceConfig, load_trace_config
from diffusers import StableDiffusionPipeline

# Parse command-line arguments for output directory
parser = argparse.ArgumentParser(description="Stable Diffusion model AOT importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save the output files.",
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/trace.toml.",
)
args = parser.parse_args()
output_dir = Path(args.output_dir).resolve()

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent
if args.trace:
    trace_text_encoder = TraceConfig(
        load_trace_config(model_dir / "trace" / "text_encoder.toml")
    )
    trace_unet = TraceConfig(load_trace_config(model_dir / "trace" / "unet.toml"))
    trace_vae = TraceConfig(load_trace_config(model_dir / "trace" / "vae.toml"))
    verbose = False
    verbose_path_text_encoder = None
    verbose_path_unet = None
    verbose_path_vae = None
else:
    trace_text_encoder = None
    trace_unet = None
    trace_vae = None
    verbose = True
    graph_dir = output_dir / "output"
    graph_dir.mkdir(parents=True, exist_ok=True)
    verbose_path_text_encoder = os.path.join(
        graph_dir, "buddy-graph-text-encoder.txt"
    )
    verbose_path_unet = os.path.join(graph_dir, "buddy-graph-unet.txt")
    verbose_path_vae = os.path.join(graph_dir, "buddy-graph-vae.txt")
    for verbose_path in (
        verbose_path_text_encoder,
        verbose_path_unet,
        verbose_path_vae,
    ):
        if os.path.exists(verbose_path):
            os.remove(verbose_path)

device = torch.device("cpu")
model_id = "borno1/stable_diffusion_2_base"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)
pipe.text_encoder.eval()
pipe.unet.eval()
pipe.vae.eval()

text_encoder = pipe.text_encoder.forward
unet = pipe.unet.forward
vae = pipe.vae.decode

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler_text_encoder = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_text_encoder",
    verbose=verbose,
    verbose_path=verbose_path_text_encoder,
    trace=trace_text_encoder,
)

dynamo_compiler_unet = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_unet",
    verbose=verbose,
    verbose_path=verbose_path_unet,
    trace=trace_unet,
)

dynamo_compiler_vae = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    func_name="forward_vae",
    verbose=verbose,
    verbose_path=verbose_path_vae,
    trace=trace_vae,
)

data_text_encoder = torch.ones((1, 77), dtype=torch.int64).to(device)
data_unet = {
    "sample": torch.ones((2, 4, 64, 64), dtype=torch.float32).to(device),
    "timestep": torch.tensor([1], dtype=torch.float32).to(device),
    "encoder_hidden_states": torch.ones((2, 77, 1024), dtype=torch.float32).to(device),
}
data_vae = torch.ones((1, 4, 64, 64), dtype=torch.float32).to(device)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs_text_encoder = dynamo_compiler_text_encoder.importer(
        text_encoder, data_text_encoder, None
    )
    graphs_unet = dynamo_compiler_unet.importer(unet, **data_unet)
    graphs_vae = dynamo_compiler_vae.importer(vae, data_vae, return_dict=False)

assert len(graphs_text_encoder) == 1
assert len(graphs_unet) == 1
assert len(graphs_vae) == 1

graph_text_encoder = graphs_text_encoder[0]
graph_unet = graphs_unet[0]
graph_vae = graphs_vae[0]

params_text_encoder = dynamo_compiler_text_encoder.imported_params[graph_text_encoder]
params_unet = dynamo_compiler_unet.imported_params[graph_unet]
params_vae = dynamo_compiler_vae.imported_params[graph_vae]

group_text_encoder = []
for op in graph_text_encoder.body:
    if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
        continue
    group_text_encoder.append(op)
graph_text_encoder.op_groups["subgraph0_text_encoder"] = group_text_encoder
graph_text_encoder.group_map_device["subgraph0_text_encoder"] = DeviceType.CPU

group_unet = []
for op in graph_unet.body:
    if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
        continue
    group_unet.append(op)
graph_unet.op_groups["subgraph0_unet"] = group_unet
graph_unet.group_map_device["subgraph0_unet"] = DeviceType.CPU

group_vae = []
for op in graph_vae.body:
    if isinstance(op, PlaceholderOp) or isinstance(op, OutputOp):
        continue
    group_vae.append(op)
graph_vae.op_groups["subgraph0_vae"] = group_vae
graph_vae.group_map_device["subgraph0_vae"] = DeviceType.CPU

driver_text_encoder = GraphDriver(graphs_text_encoder[0])
driver_unet = GraphDriver(graphs_unet[0])
driver_vae = GraphDriver(graphs_vae[0])

driver_text_encoder.subgraphs[0].lower_to_top_level_ir()
driver_unet.subgraphs[0].lower_to_top_level_ir()
driver_vae.subgraphs[0].lower_to_top_level_ir()

# Save output files to specified directory
with open(output_dir / "subgraph0_text_encoder.mlir", "w") as module_file:
    print(driver_text_encoder.subgraphs[0]._imported_module, file=module_file)
with open(output_dir / "forward_text_encoder.mlir", "w") as module_file:
    print(driver_text_encoder.construct_main_graph(True), file=module_file)

with open(output_dir / "subgraph0_unet.mlir", "w") as module_file:
    print(driver_unet.subgraphs[0]._imported_module, file=module_file)
with open(output_dir / "forward_unet.mlir", "w") as module_file:
    print(driver_unet.construct_main_graph(True), file=module_file)

with open(output_dir / "subgraph0_vae.mlir", "w") as module_file:
    print(driver_vae.subgraphs[0]._imported_module, file=module_file)
with open(output_dir / "forward_vae.mlir", "w") as module_file:
    print(driver_vae.construct_main_graph(True), file=module_file)

float32_param_text_encoder = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params_text_encoder[:-1]]
)
float32_param_text_encoder.tofile(output_dir / "arg0_text_encoder.data")

int64_param_text_encoder = params_text_encoder[-1].detach().cpu().numpy().reshape([-1])
int64_param_text_encoder.tofile(output_dir / "arg1_text_encoder.data")

param_unet = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params_unet]
)
param_unet.tofile(output_dir / "arg0_unet.data")

param_vae = numpy.concatenate(
    [param.detach().cpu().numpy().reshape([-1]) for param in params_vae]
)
param_vae.tofile(output_dir / "arg0_vae.data")
