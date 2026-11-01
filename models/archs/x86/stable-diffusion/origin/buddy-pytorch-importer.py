# ===- buddy-pytorch-importer.py ----------------------------------------------
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

from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from torch._inductor.decomposition import decompositions

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

text_encoder = import_module("models.models.stable-diffusion.model-text_encoder")
unet = import_module("models.models.stable-diffusion.model-unet")
vae = import_module("models.models.stable-diffusion.model-vae")

# Text encoder: FP32 weights and the INT64 position ids.
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    func_name="forward_text_encoder",
    trace=TraceConfig(load_trace_config(model_dir / "trace/text_encoder.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(text_encoder.load(), *text_encoder.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0_text_encoder.mlir").write_text(
    str(driver.subgraphs[0]._imported_module)
)
(output / "forward_text_encoder.mlir").write_text(
    str(driver.construct_main_graph(True))
)
np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params[:-1]]).tofile(
    output / "arg0_text_encoder.data"
)
params[-1].detach().cpu().numpy().reshape(-1).tofile(output / "arg1_text_encoder.data")

# UNet: one denoising step over the classifier-free guidance batch.
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    func_name="forward_unet",
    trace=TraceConfig(load_trace_config(model_dir / "trace/unet.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(unet.load(), *unet.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0_unet.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward_unet.mlir").write_text(str(driver.construct_main_graph(True)))
np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params]).tofile(
    output / "arg0_unet.data"
)

# VAE decoder: latents to the image.
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    func_name="forward_vae",
    trace=TraceConfig(load_trace_config(model_dir / "trace/vae.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(vae.load(), *vae.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0_vae.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward_vae.mlir").write_text(str(driver.construct_main_graph(True)))
np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in params]).tofile(
    output / "arg0_vae.data"
)
