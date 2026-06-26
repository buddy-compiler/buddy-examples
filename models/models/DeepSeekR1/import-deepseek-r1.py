# ===- import-deepseek-r1.py ---------------------------------------------------
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
# This is the test of DeepSeekR1 model.
#
# ===---------------------------------------------------------------------------

import os
import argparse
import time
from pathlib import Path
import torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.trace import TraceConfig, load_trace_config

# Add argument parser to allow custom output directory.
parser = argparse.ArgumentParser(description="DeepSeekR1 Model AOT Importer")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./",
    help="Directory to save output files.",
)
parser.add_argument(
    "--trace",
    action="store_true",
    default=False,
    help="Import with trace/trace.toml.",
)
args = parser.parse_args()

# Ensure the output directory exists.
output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
model_dir = Path(__file__).resolve().parent
if args.trace:
    trace = TraceConfig(load_trace_config(model_dir / "trace" / "trace.toml"))
    verbose = False
    verbose_path = None
else:
    trace = None
    verbose = True
    verbose_path = os.path.join(output_dir, "output", "buddy-graph.txt")
    if os.path.exists(verbose_path):
        os.remove(verbose_path)

# Retrieve the DeepSeekR1 model path from environment variables.
model_path = os.environ.get("DEEPSEEKR1_MODEL_PATH")
if model_path is None:
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Initialize the model from the specified model path.
model = AutoModelForCausalLM.from_pretrained(model_path).eval()
model.config.use_cache = False

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
    verbose=verbose,
    verbose_path=verbose_path,
    trace=trace,
)

# Import the model into MLIR module and parameters.
with torch.no_grad():
    data = {
        "input_ids": torch.zeros((1, 40), dtype=torch.int64),
        "attention_mask": torch.zeros((1, 40), dtype=torch.int64),
    }
    graphs = dynamo_compiler.importer(
        model,
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
    )

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()


def tensor_to_numpy(param):
  param = param.detach().cpu().contiguous()
  if param.dtype == torch.bfloat16:
    return param.view(torch.uint16).numpy().reshape([-1])
  if param.dtype == torch.float32:
    return param.numpy().reshape([-1])
  raise TypeError(f"Unsupported DeepSeekR1 parameter dtype: {param.dtype}")


# Save the generated files to the specified output directory.
with open(output_dir / "subgraph0.mlir", "w") as module_file:
  print(driver.subgraphs[0]._imported_module, file=module_file)
with open(output_dir / "forward.mlir", "w") as module_file:
  print(driver.construct_main_graph(True), file=module_file)
all_param = numpy.concatenate([tensor_to_numpy(param) for param in params])
all_param.tofile(output_dir / "arg0.data")
