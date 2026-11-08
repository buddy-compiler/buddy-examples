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
import json

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from compiler.frontend import quant
from buddy.compiler.trace import TraceConfig, load_trace_config
from torch._inductor.decomposition import decompositions
from diffusers.models.attention_processor import Attention, AttnProcessor
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(parents=True, exist_ok=True)

text_encoder = import_module("models.models.stable-diffusion.model-text_encoder")
unet = import_module("models.models.stable-diffusion.model-unet")
vae = import_module("models.models.stable-diffusion.model-vae")


def attention_scores(self, query, key, attention_mask=None):
    """The attention probabilities from one batched product, scaled: the score
    product is then a BatchMatmul the quantization reads like every other
    product, not the baddbmm diffusers fuses the scale into."""
    scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
    if attention_mask is not None:
        scores = scores + attention_mask
    return scores.softmax(dim=-1)


Attention.get_attention_scores = attention_scores
# The activation scales are computed online; buddy-pytorch-calibrate.py
# accepts the quantized graphs first.
calibration = json.loads((output / "calibration.json").read_text())
print(f"Quantized image accepted at {calibration['psnr']:.2f} dB PSNR")

# Text encoder: FP32 weights and the INT64 position ids.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    func_name="forward_text_encoder",
    trace=TraceConfig(load_trace_config(model_dir / "trace/text_encoder.toml")),
)
model = text_encoder.load()
# The attention runs through explicit products the quantization can read as
# INT8, not through one fused attention.
model.text_encoder.config._attn_implementation = "eager"
with torch.no_grad():
    graphs = compiler.importer(model, *text_encoder.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.dump(output / "01-import-text-encoder.txt")
# Absorb the transposes PyTorch puts on the Linear weights.
permute(graph)
graph.dump(output / "02-permuted-text-encoder.txt")
# Every matrix product reads quantized operands: MXINT8 for the Linears and
# the attention products, INT8 for the convolutions.
params = quantize(graph, params)
graph.dump(output / "03-quanted-text-encoder.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0_text_encoder.mlir").write_text(
    str(driver.subgraphs[0]._imported_module)
)
(output / "forward_text_encoder.mlir").write_text(
    str(driver.construct_main_graph(True))
)
# forward takes the parameters grouped by type: FP32 (the biases among
# them), INT8 weights (MXINT8 codes and exponents for the Linears), then
# the INT64 position ids.
for index, dtype in enumerate((torch.float32, torch.int8, torch.int64)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}_text_encoder.data")

# UNet: one denoising step over the classifier-free guidance batch.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    func_name="forward_unet",
    trace=TraceConfig(load_trace_config(model_dir / "trace/unet.toml")),
)
model = unet.load()
model.unet.set_attn_processor(AttnProcessor())
with torch.no_grad():
    graphs = compiler.importer(model, *unet.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.dump(output / "01-import-unet.txt")
# The convolutions run in NHWC; the transposes around the Linears are absorbed.
permute(graph)
graph.dump(output / "02-permuted-unet.txt")
params = quantize(graph, params)
graph.dump(output / "03-quanted-unet.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0_unet.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward_unet.mlir").write_text(str(driver.construct_main_graph(True)))
for index, dtype in enumerate((torch.float32, torch.int8)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}_unet.data")

# VAE decoder: latents to the image.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    func_name="forward_vae",
    trace=TraceConfig(load_trace_config(model_dir / "trace/vae.toml")),
)
model = vae.load()
model.vae.set_attn_processor(AttnProcessor())
with torch.no_grad():
    graphs = compiler.importer(model, *vae.inputs())
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.dump(output / "01-import-vae.txt")
# The convolutions run in NHWC; the transposes around the Linears are absorbed.
permute(graph)
graph.dump(output / "02-permuted-vae.txt")
params = quantize(graph, params)
graph.dump(output / "03-quanted-vae.txt")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0_vae.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward_vae.mlir").write_text(str(driver.construct_main_graph(True)))
for index, dtype in enumerate((torch.float32, torch.int8)):
    np.concatenate(
        [p.detach().cpu().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}_vae.data")
