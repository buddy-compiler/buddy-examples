import sys
from pathlib import Path

#!/usr/bin/env python3
# ===- buddy-pytorch-importer.py ------------------------------------------
#
# MiniMax-H3 Base Ref2VA AOT importer (text_encoder / transformer /
# visual_vae / audio_vae). Same layout as import-stable-diffusion.py.
#
# ===---------------------------------------------------------------------------

import os
from pathlib import Path

from importlib import import_module

parameters = {
    "duration_s": 4,
    "fps": 24,
    "height": 768,
    "width": 1344,
    "latent_c": 24,
    "audio_latent_c": 32,
    "text_dim": 5120,
    "text_len": 512,
    "sample_rate": 32000,
    "denoise_steps": 30,
    "video_shift": 12.0,
}
parameters["frames"] = parameters["duration_s"] * parameters["fps"]
parameters["latent_t"] = parameters["frames"] // 4
parameters["latent_h"] = parameters["height"] // 16
parameters["latent_w"] = parameters["width"] // 16
parameters["audio_latent_t"] = parameters["duration_s"] * 40


if parameters["width"] % 32 != 0 or parameters["height"] % 32 != 0:
    raise ValueError(
        f"spatial {parameters["width"]}x{parameters["height"]} must be divisible by 32"
    )

model_path = Path(__file__).resolve().parent / "output/minimax-h3-ref2va"

import numpy
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from diffusers import DiffusionPipeline
from torch._inductor.decomposition import decompositions as inductor_decomp

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent
trace_dir = model_dir / "trace"

device = torch.device("cpu")

pipe = DiffusionPipeline.from_pretrained(
    model_path / "Ref2VA", trust_remote_code=True, torch_dtype=torch.float32
).to(device)
for name in ("text_encoder", "transformer", "video_vae", "audio_vae"):
    getattr(pipe, name).eval()

text_encoder = pipe.text_encoder
transformer = pipe.transformer
visual_vae = pipe.video_vae
audio_vae = pipe.audio_vae


def export_graph(fn, example_args, name, **example_kwargs):
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        verbose=True,
        verbose_path=output_dir / f"buddy-graph-{name}.txt",
        trace=None,
    )
    with torch.no_grad():
        graphs = compiler.importer(fn, *example_args, **example_kwargs)
    for graph in graphs:
        graph.lower_to_top_level_ir()


# Example tensors. Wrong shapes/kwargs fail here on purpose.
data_visual = torch.ones(
    (
        1,
        parameters["latent_c"],
        parameters["latent_t"],
        parameters["latent_h"],
        parameters["latent_w"],
    ),
    dtype=torch.float32,
    device=device,
)
data_audio = torch.ones(
    (1, parameters["audio_latent_c"], parameters["audio_latent_t"]),
    dtype=torch.float32,
    device=device,
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


export_graph(
    _text_fwd,
    (),
    "text_encoder",
    input_ids=torch.ones((1, parameters["text_len"]), dtype=torch.int64, device=device),
    attention_mask=torch.ones(
        (1, parameters["text_len"]), dtype=torch.int64, device=device
    ),
)

export_graph(
    transformer.forward,
    (),
    "transformer",
    hidden_states=torch.ones(
        (
            1,
            parameters["latent_c"],
            parameters["latent_t"],
            parameters["latent_h"],
            parameters["latent_w"],
        ),
        dtype=torch.float32,
        device=device,
    ),
    timestep=torch.tensor([0.5], dtype=torch.float32, device=device),
    encoder_hidden_states=torch.ones(
        (1, parameters["text_len"], parameters["text_dim"]),
        dtype=torch.float32,
        device=device,
    ),
    audio_hidden_states=torch.ones(
        (1, parameters["audio_latent_c"], parameters["audio_latent_t"]),
        dtype=torch.float32,
        device=device,
    ),
)

# Decode owns arg0_visual_vae.data (weights). Encode is a second graph.
export_graph(_visual_decode, (data_visual,), "visual_vae")

data_frames = torch.ones(
    (1, 3, parameters["frames"], parameters["height"], parameters["width"]),
    dtype=torch.float32,
    device=device,
)
export_graph(_visual_encode, (data_frames,), "visual_vae_encode")

export_graph(_audio_decode, (data_audio,), "audio_vae")

(output_dir / "buddy-graph.txt").write_text(
    "\n".join(
        (output_dir / f"buddy-graph-{name}.txt").read_text()
        for name in (
            "text_encoder",
            "transformer",
            "visual_vae",
            "visual_vae_encode",
            "audio_vae",
        )
    )
)
