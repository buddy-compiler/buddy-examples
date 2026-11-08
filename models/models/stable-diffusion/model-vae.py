# ===- model-vae.py ------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Stable Diffusion VAE decoder graph: latents to the image.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKL

checkpoint = Path(__file__).resolve().parent / "output/stable-diffusion"


class VAE(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents, return_dict=False)


def load():
    """The VAE of the pipeline, ready to import."""
    vae = AutoencoderKL.from_pretrained(checkpoint / "vae", torch_dtype=torch.float32)
    return VAE(vae).eval()


def inputs():
    """Example inputs of the graph: the latents of one image."""
    return (torch.ones((1, 4, 64, 64), dtype=torch.float32),)
