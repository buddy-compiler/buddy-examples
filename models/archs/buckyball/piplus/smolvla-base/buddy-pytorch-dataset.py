# ===- buddy-pytorch-dataset.py --------------------------------------------
#
# SmolVLA on piplus: the dataset of the accuracy runs (dataset.cpp), the
# 100th frame of each of the 50 SO100 pick-and-place episodes (one frame an
# episode: a whole policy takes minutes of BEMU, the accuracy run of the
# design stays within two hours), prepared as observation.py prepares it. In
# <output>/dataset/:
#   observations.data  each frame as the program takes it (model.h's
#                      Observation)
#   reference.data     the actions of the FP32 program of the graphs on it,
#                      as baremetal.cpp checks them
#   truth.data         the 50 actions the episode recorded from the frame on,
#                      normalized as the policy's outputs (the mean and
#                      deviation of the dataset), 50 x 6 floats
#   labels.data        the episode, int64
#   model/ddr/         the parameters, which every run reads from the region
# The reference is computed by 10 forked processes of 4 threads.
#
#   buddy-pytorch-dataset.py <output>
#
# ===---------------------------------------------------------------------===

from importlib import import_module
from pathlib import Path
import json
import multiprocessing
import sys

import numpy as np
import torch

from frontend import program
from frontend.graphs import Prefix, Step, Vision

output = Path(sys.argv[1])
dataset = output / "dataset"
(dataset / "model/ddr").mkdir(parents=True, exist_ok=True)
# One thread here: a forked worker would wait forever at the barrier of an
# OpenMP team this process had started.
torch.set_num_threads(1)
model = import_module("models.models.smolvla-base.model")
observation = import_module("models.models.smolvla-base.observation")
policy = model.load()
policy.flow.vlm_with_expert.get_vlm_model().vision_model.config._attn_implementation = (
    "eager"
)
flow = policy.flow
vision, prefix, step = Vision(flow).eval(), Prefix(flow).eval(), Step(flow).eval()
steps = flow.config.num_steps
chunk = flow.config.chunk_size
statistics = json.loads((observation.root / "meta/stats.json").read_text())["action"]
mean = torch.tensor(statistics["mean"])
std = torch.tensor(statistics["std"])
episodes = len(observation.starts)


def work(episode):
    """The frame of `episode` as the program takes it, the actions of the
    FP32 program on it and the recorded actions from it on."""
    torch.set_num_threads(4)
    frame = observation.frame(episode)
    with torch.no_grad():
        actions = program.run(vision, prefix, step, steps, *frame)
    first = observation.starts[episode] + 100
    truth = torch.stack(
        [observation.dataset[first + k]["action"] for k in range(chunk)]
    )
    assert observation.dataset[first + chunk - 1]["episode_index"] == episode
    return (
        b"".join(value.numpy().tobytes() for value in frame),
        actions.numpy().astype(np.float32).tobytes(),
        ((truth - mean) / std).numpy().astype(np.float32).tobytes(),
    )


with multiprocessing.get_context("fork").Pool(10) as pool:
    results = pool.map(work, range(episodes))
(dataset / "observations.data").write_bytes(b"".join(r[0] for r in results))
(dataset / "reference.data").write_bytes(b"".join(r[1] for r in results))
(dataset / "truth.data").write_bytes(b"".join(r[2] for r in results))
np.arange(episodes, dtype=np.int64).tofile(dataset / "labels.data")
for name in (
    "vision-arg0.data",
    "vision-arg1.data",
    "prefix-pack-f32.data",
    "prefix-pack-i8.data",
    "step-pack-f32.data",
    "step-pack-i8.data",
):
    link = dataset / "model/ddr" / name
    link.unlink(missing_ok=True)
    link.symlink_to(f"../../../{name}")
print(f"{episodes} SO100 frames for the accuracy runs")
