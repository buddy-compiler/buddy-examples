# ===- buddy-pytorch-importer.py -------------------------------------------------------
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

from pathlib import Path

import numpy
import torch
import torch.nn as nn
import transformers.masking_utils as masking_utils
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import WhisperForConditionalGeneration, AutoTokenizer

output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
masking_utils.find_packed_sequence_indices = lambda *args, **kwargs: None
model = WhisperForConditionalGeneration.from_pretrained(
    Path(__file__).resolve().parents[4] / "models/whisper/output/whisper"
)
model.config.use_cache = False
model.eval()


class Logits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        return self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits


input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)
decoder_input_ids = torch.zeros((1, 448), dtype=torch.long)
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)
with torch.no_grad():
    graphs = dynamo_compiler.importer(
        Logits(model),
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
    )
if len(graphs) != 1:
    raise RuntimeError(f"Expected one graph, got {len(graphs)}")
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
with open(output_dir / "subgraph0.mlir", "w") as f:
    print(driver.subgraphs[0]._imported_module, file=f)
with open(output_dir / "forward.mlir", "w") as f:
    print(driver.construct_main_graph(True), file=f)
numpy.concatenate([p.detach().numpy().reshape([-1]) for p in params]).tofile(
    output_dir / "arg0.data"
)
print(f"[import-whisper] Wrote forward.mlir, subgraph0.mlir, arg0.data → {output_dir}")

tokenizer = AutoTokenizer.from_pretrained(
    Path(__file__).resolve().parents[4] / "models/whisper/output/whisper"
)
with (output_dir / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
