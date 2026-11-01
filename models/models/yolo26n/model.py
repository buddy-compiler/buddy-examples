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
# YOLO26n model definition.
#
# ===---------------------------------------------------------------------------

from pathlib import Path

import torch
from ultralytics import YOLO

checkpoint = Path(__file__).resolve().parent / "output/yolo26n"


def load():
    """Ultralytics YOLO26n with the COCO weights, ready to import: the Detect
    head exports its end-to-end boxes as [x1, y1, x2, y2]."""
    model = YOLO(checkpoint / "yolo26n.pt").model.eval()
    head = model.model[-1]
    head.end2end = True
    head.export = True
    head.xyxy = True
    with torch.no_grad():
        # Populate the detection anchors before capture so the cache updates
        # are not graph outputs.
        model(*inputs())
    return model


def inputs():
    """Example inputs of the graph: one 640x640 RGB image."""
    return (torch.zeros((1, 3, 640, 640)),)
