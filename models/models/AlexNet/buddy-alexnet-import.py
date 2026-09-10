# ===- buddy-alexnet-import.py ---------------------------------------------
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
# This is the AlexNet model AOT importer.
#
# Model: https://huggingface.co/Pie33000/alexnet (from-scratch AlexNet,
# ImageNet-1K, epoch-40 checkpoint `model_40.pth`).
# Original implementation: https://github.com/pie33000/alexnet (`model.py`).
#
# ===---------------------------------------------------------------------------

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch._inductor.lowering
from torch._inductor.decomposition import decompositions as inductor_decomp
from torch._decomp import remove_decompositions

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa

# AlexNet architecture, identical to the author's `model.py`
# (https://github.com/pie33000/alexnet): 5 conv layers (96/256/384/384/256),
# max-pool 2x2 after conv1/conv2, no LRN, then 3 fully-connected layers
# (43264 -> 4096 -> 4096 -> 1000) with dropout in between.
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 13 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Checkpoint on HuggingFace.
CKPT_URL = "https://huggingface.co/Pie33000/alexnet/resolve/main/model_40.pth"
CKPT_SHA256 = "265ca8ad8b877666f1b8c22f612fb1a2"


def ensure_checkpoint(path: Path) -> None:
    """Download model_40.pth from HuggingFace when it is not cached locally."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Import] Downloading {CKPT_URL} -> {path}")
    urllib.request.urlretrieve(CKPT_URL, path)
    if not path.is_file():
        raise RuntimeError(f"checkpoint download failed: {path}")
    print(f"[Import] checkpoint ready: {path}")


def fix_maxpool_floor_semantics(mlir_text: str) -> str:
    """Work around a buddy-mlir TOSA max_pool2d lowering bug.

    The TOSA dialect requires (H + pad_top + pad_bottom - kernel) to be wholly
    divisible by stride. For AlexNet's second max-pool (2x2, stride 2 on a
    27x27 map) torch's floor semantics give 13x13, which TOSA cannot express
    with any padding ((27 + p - 2) % 2 == 0 forces p odd -> 14x14). The buddy
    importer emits a padded 14x14 pool but keeps 13x13 downstream types, which
    the TOSA verifier rejects. We re-emit the padded pool followed by a
    `tosa.slice` back to the torch floor size (same pad+slice trick the
    importer already uses for conv1).
    """
    import re

    lines = mlir_text.splitlines()
    out: list[str] = []
    pending = None  # (pool_result, pool_out_type, torch_h, torch_w, out_c)
    for line in lines:
        # Detect a padded max_pool2d whose output exceeds torch's floor size.
        m = re.search(
            r'%(\w+) = "tosa.max_pool2d"\(%(?:\w+)\) <\{kernel = array<i64: (\d+), (\d+)>,'
            r'.*?pad = array<i64: (\d+), (\d+), (\d+), (\d+)>.*?stride = array<i64: (\d+), (\d+)>'
            r'\}> : \(tensor<1x(\d+)x(\d+)x(\d+)xf32>\) -> tensor<1x(\d+)x(\d+)x(\d+)xf32>',
            line,
        )
        if m:
            name, kh, kw, pt, pb, pl, pr, sh, sw, in_h, in_w, in_c, out_h, out_w, out_c = (
                m.groups()
            )
            kh = int(kh); kw = int(kw); sh = int(sh); sw = int(sw)
            pt = int(pt); pb = int(pb); pl = int(pl); pr = int(pr)
            in_h = int(in_h); in_w = int(in_w); in_c = int(in_c)
            out_h = int(out_h); out_w = int(out_w); out_c = int(out_c)
            torch_h = (in_h - kh) // sh + 1
            torch_w = (in_w - kw) // sw + 1
            if (torch_h, torch_w) != (out_h, out_w):
                # The padded TOSA pool overshoots torch's floor semantics;
                # keep the padded pool and slice back to the torch size.
                pool_out_type = f"tensor<1x{out_h}x{out_w}x{out_c}xf32>"
                slice_type = f"tensor<1x{torch_h}x{torch_w}x{out_c}xf32>"
                out.append(line)
                out.append(
                    f'    %slice_start_{name} = "tosa.const_shape"() '
                    f'<{{values = dense<0> : tensor<4xindex>}}> : () -> !tosa.shape<4>'
                )
                out.append(
                    f'    %slice_size_{name} = "tosa.const_shape"() '
                    f'<{{values = dense<[1, {torch_h}, {torch_w}, {out_c}]> : tensor<4xindex>}}> '
                    f': () -> !tosa.shape<4>'
                )
                out.append(
                    f'    %slice_{name} = "tosa.slice"(%{name}, %slice_start_{name}, '
                    f'%slice_size_{name}) : ({pool_out_type}, !tosa.shape<4>, !tosa.shape<4>) '
                    f'-> {slice_type}'
                )
                pending = (name, pool_out_type, torch_h, torch_w, out_c)
                continue
        out.append(line)

    # Rewire the first consumer of the padded pool (a later line, never the
    # inserted slice itself) to the slice result.
    if pending is not None:
        name, pool_out_type, torch_h, torch_w, out_c = pending
        slice_type = f"tensor<1x{torch_h}x{torch_w}x{out_c}xf32>"
        rewired = False
        for i, line in enumerate(out):
            if rewired:
                break
            if f'%slice_{name} =' in line:
                continue
            if f'(%{name},' in line:
                line = line.replace(f'(%{name},', f'(%slice_{name},', 1)
                line = line.replace(f'({pool_out_type},', f'({slice_type},', 1)
                out[i] = line
                rewired = True
        assert rewired, "failed to locate max_pool2d consumer"
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="AlexNet model AOT importer")
    parser.add_argument(
        "--output-dir", type=str, default="./", help="Directory to save output files."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model_40.pth (downloaded from HuggingFace when omitted).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(__file__).resolve().parent

    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else model_dir / "model_40.pth"
    )
    ensure_checkpoint(checkpoint)

    # Build the model and load the epoch-40 checkpoint.
    model = AlexNet()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()
    print(f"[Import] checkpoint loaded: {checkpoint}")
    print(f"[Import] params: {sum(p.numel() for p in model.parameters())}")

    DEFAULT_DECOMPOSITIONS = [
        torch.ops.aten.max_pool2d_with_indices.default,
    ]
    remove_decompositions(inductor_decomp, DEFAULT_DECOMPOSITIONS)

    # Initialize Dynamo Compiler with specific configurations as an importer.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    data = torch.randn([1, 3, 224, 224])
    # Import the model into MLIR module and parameters.
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, data)
    assert len(graphs) == 1
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]
    pattern_list = [simply_fuse]
    graphs[0].fuse_ops(pattern_list)
    driver = GraphDriver(graphs[0])
    driver.subgraphs[0].lower_to_top_level_ir()

    # Write the MLIR module and forward graph to the specified output directory.
    subgraph0_text = str(driver.subgraphs[0]._imported_module)
    subgraph0_text = fix_maxpool_floor_semantics(subgraph0_text)
    with open(output_dir / "subgraph0.mlir", "w") as module_file:
        print(subgraph0_text, file=module_file)
    with open(output_dir / "forward.mlir", "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)

    params = dynamo_compiler.imported_params[graph]
    float32_param = np.concatenate(
        [
            param.detach().numpy().reshape([-1])
            for param in params
            if param.dtype == torch.float32
        ]
    )
    float32_param.tofile(output_dir / "arg0.data")
    print(f"[Import] arg0.data floats: {float32_param.size}")


if __name__ == "__main__":
    main()
