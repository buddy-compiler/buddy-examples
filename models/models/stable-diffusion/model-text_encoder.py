# ===- model-text_encoder.py ---------------------------------------------------
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
# Stable Diffusion text encoder graph: prompt tokens to the CLIP hidden states.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.nn as nn
from transformers import CLIPTextModel

checkpoint = Path(__file__).resolve().parent / "output/stable-diffusion"


class TextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids, None)


def load():
    """The CLIP text encoder of the pipeline, ready to import."""
    text_encoder = CLIPTextModel.from_pretrained(
        checkpoint / "text_encoder", torch_dtype=torch.float32
    )
    return TextEncoder(text_encoder).eval()


def inputs():
    """Example inputs of the graph: one prompt of 77 token ids."""
    return (torch.ones((1, 77), dtype=torch.int64),)
