from pathlib import Path
import json

import torch
import tomllib
from quant.calibration import WikiTextCalibration
from torch._inductor.decomposition import decompositions
from transformers import AutoTokenizer, AutoModelForCausalLM

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from stack.compiler.frontend.quant.native_mxfp8 import (
    Cache,
    calibrate,
    prepare,
    register,
    scale_for,
)

model_dir = Path(__file__).resolve().parent
output = model_dir / "output"
output.mkdir(exist_ok=True)
checkpoint = (
    model_dir
    / "../../../models/deepseek-r1-distill-qwen-1.5b/output/deepseek-r1-distill-qwen-1.5b"
)

# ===----------------------------------------------------------------------===
# Pack weights and calibrate fixed activation and KV scales on training text.
# ===----------------------------------------------------------------------===
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, dtype=torch.float32, attn_implementation="eager"
).eval()
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
embedding_scales = {}
prepare(model, embedding_scales=embedding_scales)
config = tomllib.loads((model_dir / "quant/quant.toml").read_text())
capacity = config["cache"]["capacity"]
activation_scales, calibration_cache = calibrate(
    model, WikiTextCalibration(tokenizer, capacity), **config["calibration"]
)
(output / "calibration.json").write_text(
    json.dumps(
        {
            "dataset": "Salesforce/wikitext",
            "configuration": "wikitext-2-raw-v1",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "split": "train",
            "sequence_length": capacity,
            "seed": 0,
            **config["calibration"],
        },
        indent=2,
    )
    + "\n"
)
(output / "activation-scales.json").write_text(
    json.dumps(activation_scales, indent=2) + "\n"
)


# ===----------------------------------------------------------------------===
# Persist only MXFP8 KV bytes between token steps; scales are packed parameters.
# ===----------------------------------------------------------------------===
class Decoder(torch.nn.Module):
    def __init__(self, model, maxima, centers):
        super().__init__()
        self.model = model
        self.capacity = capacity
        self.layouts = {}
        self.cache_size = 0
        for layer, maximum in sorted(maxima.items()):
            _, heads, blocks = maximum.shape
            width = blocks * 32
            self.layouts[layer] = (self.cache_size, heads, width)
            self.cache_size += 2 * heads * self.capacity * width
            self.register_buffer(f"kv_scale_{layer}", scale_for(maximum))
            self.register_buffer(f"key_center_{layer}", centers[layer].flatten())

    def forward(self, tokens, positions, packed_cache):
        scales = {layer: getattr(self, f"kv_scale_{layer}") for layer in self.layouts}
        centers = {
            layer: getattr(self, f"key_center_{layer}") for layer in self.layouts
        }
        cache = Cache(
            packed_cache, positions, self.layouts, scales, centers, self.capacity
        )
        visible = (
            torch.arange(self.capacity)[None, None, None, :]
            <= positions[:, None, :, None]
        )
        mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min)
        logits = self.model(
            input_ids=tokens,
            position_ids=positions,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            past_key_values=cache,
            use_cache=True,
        ).logits
        return logits, torch.cat(cache.updated)


maxima, centers = calibration_cache.parameters()
decoder = Decoder(model, maxima, centers).eval()
(output / "cache.json").write_text(
    json.dumps(
        {
            "capacity": capacity,
            "code_bytes": decoder.cache_size,
            "layouts": decoder.layouts,
            "centers": {
                str(layer): center.tolist() for layer, center in centers.items()
            },
            "scales": {
                str(layer): (
                    getattr(decoder, f"kv_scale_{layer}").to(torch.int16) & 255
                ).tolist()
                for layer in decoder.layouts
            },
        },
        indent=2,
    )
    + "\n"
)
(output / "cache.h").write_text(
    f"#pragma once\nconstexpr size_t CacheBytes = {decoder.cache_size};\nconstexpr size_t CacheCapacity = {capacity};\n"
)

# ===----------------------------------------------------------------------===
# Capture the quantized graph and preserve its packed parameter ABI.
# ===----------------------------------------------------------------------===
compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry, aot_autograd_decomposition=decompositions
)
register(compiler)
inputs = (
    torch.zeros((1, 1), dtype=torch.int64),
    torch.zeros((1, 1), dtype=torch.int64),
    torch.zeros(decoder.cache_size, dtype=torch.int8),
)
with torch.no_grad():
    (graph,) = compiler.importer(decoder, *inputs)
graph.order_inputs(inputs)
params = compiler.imported_params[graph]
graph.dump(output / "02-quanted.txt")
# graph.dump_parameters(output / "02-quanted-parameters")
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
for subgraph in driver.subgraphs:
    subgraph.trace(model_dir / "trace/trace.toml")
graph.dump(output / "04-traced.txt")
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir()
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))

for index, dtype in enumerate((torch.float32, torch.int8)):
    with (output / f"arg{index}.data").open("wb") as stream:
        for param in params:
            if param.dtype == dtype:
                stream.write(param.detach().cpu().contiguous().numpy().tobytes())
assert {param.dtype for param in params} == {torch.float32, torch.int8}
with (output / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
print(
    "Exported MXFP8 weights, fixed activation/KV scales and persistent INT8 KV cache",
    flush=True,
)
