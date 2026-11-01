# ===- model.py ----------------------------------------------------------------
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
# MobileNetV3-Small model definition.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

checkpoint = Path(__file__).resolve().parent / "output/mobilenet-v3-small"


def load():
    """torchvision MobileNetV3-Small with the ImageNet weights, ready to
    import."""
    model = mobilenet_v3_small(weights=None).eval()
    model.load_state_dict(torch.load(checkpoint / "model.pth", weights_only=True))
    for layer in model.modules():
        if isinstance(layer, (torch.nn.ReLU, torch.nn.Hardswish)):
            layer.inplace = False
    return model


def inputs():
    """Example inputs of the graph: one 224x224 RGB image."""
    return (torch.zeros((1, 3, 224, 224)),)
