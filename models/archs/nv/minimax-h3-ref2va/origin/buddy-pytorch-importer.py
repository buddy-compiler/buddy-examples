from pathlib import Path

repo_root = Path(__file__).resolve().parents[5]
compiler_build = repo_root / "thirdparty/buddy-mlir/build"

#!/usr/bin/env python3
# ===- buddy-pytorch-importer.py ------------------------------------------
#
# MiniMax-H3 Base Ref2VA AOT importer (text_encoder / transformer /
# visual_vae / audio_vae). Same layout as import-stable-diffusion.py.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path

import argparse
from importlib import import_module

parser = argparse.ArgumentParser(description="MiniMax-H3 Ref2VA AOT importer")

parser.add_argument("--duration-s", type=int, required=True)
parser.add_argument("--fps", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--latent-c", type=int, required=True)
parser.add_argument("--audio-latent-c", type=int, required=True)
parser.add_argument("--text-dim", type=int, required=True)
parser.add_argument("--text-len", type=int, required=True)
parser.add_argument("--sample-rate", type=int, required=True)
parser.add_argument("--denoise-steps", type=int, required=True)
parser.add_argument("--video-shift", type=float, required=True)
parser.add_argument("--frames", type=int, required=True)
parser.add_argument("--latent-t", type=int, required=True)
parser.add_argument("--latent-h", type=int, required=True)
parser.add_argument("--latent-w", type=int, required=True)
parser.add_argument("--audio-latent-t", type=int, required=True)

args = parser.parse_args()


if args.width % 32 != 0 or args.height % 32 != 0:
    raise ValueError(f"spatial {args.width}x{args.height} must be divisible by 32")

model_path = "MiniMaxAI/MiniMax-H3"

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.operation import OutputOp, PlaceholderOp
from buddy.compiler.graph.transform.trace import TraceInsertionPass
from buddy.compiler.graph.type import DeviceType
from stack.compiler.frontend.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from diffusers import DiffusionPipeline
from torch._inductor.decomposition import decompositions as inductor_decomp

output_dir = Path(Path("output")).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent
trace_dir = model_dir / "trace"

device = torch.device("cpu")

load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32}
local = Path(model_path)
if local.exists():
    if (local / "Ref2VA" / "model_index.json").is_file():
        local = local / "Ref2VA"
    if not (local / "model_index.json").is_file():
        raise FileNotFoundError(f"missing model_index.json under {local}")
    pipe = DiffusionPipeline.from_pretrained(str(local), **load_kwargs)
else:
    pipe = DiffusionPipeline.from_pretrained(
        model_path, subfolder="Ref2VA", **load_kwargs
    )
pipe = pipe.to(device)
for name in ("text_encoder", "transformer", "video_vae", "audio_vae"):
    if not hasattr(pipe, name):
        raise RuntimeError(f"pipeline missing component: {name}")
    getattr(pipe, name).eval()

text_encoder = pipe.text_encoder
transformer = pipe.transformer
visual_vae = pipe.video_vae
audio_vae = pipe.audio_vae


def _import_one(
    fn, example_args, func_name, subgraph_name, toml_name, **example_kwargs
):
    verbose = False
    verbose_path = None

    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        func_name=func_name,
        verbose=verbose,
        verbose_path=verbose_path,
        trace=None,
    )
    with torch.no_grad():
        graphs = compiler.importer(fn, *example_args, **example_kwargs)
    assert len(graphs) == 1, f"{func_name}: expected 1 graph, got {len(graphs)}"
    graph = graphs[0]
    params = compiler.imported_params[graph]

    toml_path = trace_dir / toml_name
    TraceInsertionPass(TraceConfig(load_trace_config(toml_path)))(graph)

    group = [op for op in graph.body if not isinstance(op, (PlaceholderOp, OutputOp))]
    graph.op_groups[subgraph_name] = group
    graph.group_map_device[subgraph_name] = DeviceType.CPU

    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    with open(output_dir / f"{subgraph_name}.mlir", "w") as f:
        print(driver.subgraphs[0]._imported_module, file=f)
    with open(output_dir / f"{func_name}.mlir", "w") as f:
        print(driver.construct_main_graph(True), file=f)

    flat = numpy.concatenate(
        [p.detach().cpu().float().numpy().reshape([-1]) for p in params]
    )
    return flat


# Example tensors. Wrong shapes/kwargs fail here on purpose.
data_visual = torch.ones(
    (1, args.latent_c, args.latent_t, args.latent_h, args.latent_w),
    dtype=torch.float32,
    device=device,
)
data_audio = torch.ones(
    (1, args.audio_latent_c, args.audio_latent_t), dtype=torch.float32, device=device
)


def _text_fwd(input_ids, attention_mask):
    out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        # H3 uses unnormalized layer-50 states when available.
        hs = out.hidden_states
        if len(hs) > 50:
            return hs[50]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    raise RuntimeError(f"unsupported text_encoder output: {type(out)}")


def _vae_out(out):
    if hasattr(out, "sample"):
        return out.sample
    if hasattr(out, "latent_dist"):
        return out.latent_dist.mode()
    if torch.is_tensor(out):
        return out
    raise RuntimeError(f"unsupported VAE output: {type(out)}")


def _visual_decode(latents):
    return _vae_out(visual_vae.decode(latents))


def _visual_encode(frames):
    return _vae_out(visual_vae.encode(frames))


def _audio_decode(latents):
    return _vae_out(audio_vae.decode(latents))


param_text = _import_one(
    _text_fwd,
    (),
    "forward_text_encoder",
    "subgraph0_text_encoder",
    "text_encoder.toml",
    input_ids=torch.ones((1, args.text_len), dtype=torch.int64, device=device),
    attention_mask=torch.ones((1, args.text_len), dtype=torch.int64, device=device),
)
param_text.tofile(output_dir / "arg0_text_encoder.data")

param_tr = _import_one(
    transformer.forward,
    (),
    "forward_transformer",
    "subgraph0_transformer",
    "transformer.toml",
    hidden_states=torch.ones(
        (1, args.latent_c, args.latent_t, args.latent_h, args.latent_w),
        dtype=torch.float32,
        device=device,
    ),
    timestep=torch.tensor([0.5], dtype=torch.float32, device=device),
    encoder_hidden_states=torch.ones(
        (1, args.text_len, args.text_dim), dtype=torch.float32, device=device
    ),
    audio_hidden_states=torch.ones(
        (1, args.audio_latent_c, args.audio_latent_t),
        dtype=torch.float32,
        device=device,
    ),
)
param_tr.tofile(output_dir / "arg0_transformer.data")

# Decode owns arg0_visual_vae.data (weights). Encode is a second graph.
param_v = _import_one(
    _visual_decode,
    (data_visual,),
    "forward_visual_vae",
    "subgraph0_visual_vae",
    "visual_vae.toml",
)
param_v.tofile(output_dir / "arg0_visual_vae.data")

data_frames = torch.ones(
    (1, 3, args.frames, args.height, args.width), dtype=torch.float32, device=device
)
param_ve = _import_one(
    _visual_encode,
    (data_frames,),
    "forward_visual_vae_encode",
    "subgraph0_visual_vae_encode",
    "visual_vae_encode.toml",
)
param_ve.tofile(output_dir / "arg0_visual_vae_encode.data")

param_a = _import_one(
    _audio_decode,
    (data_audio,),
    "forward_audio_vae",
    "subgraph0_audio_vae",
    "audio_vae.toml",
)
param_a.tofile(output_dir / "arg0_audio_vae.data")
