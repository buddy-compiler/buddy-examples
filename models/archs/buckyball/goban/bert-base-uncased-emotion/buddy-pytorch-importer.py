from importlib import import_module
from pathlib import Path
import sys
import json

import numpy as np

import torch

# A share of the host: the default takes every core of a loaded host.
torch.set_num_threads(8)
from torch._inductor.decomposition import decompositions
from transformers import BertTokenizer

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config
from compiler.frontend import quant
from frontend.permute.graph import permute
from frontend.quant.graph import quantize

model_dir = Path(__file__).resolve().parent
output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)

# ===----------------------------------------------------------------------===
# Step 1: Import the graph of the checkpoint.
# ===----------------------------------------------------------------------===
bert = import_module("models.models.bert-base-uncased-emotion.model")
model = bert.load()
inputs = bert.inputs()
# The traced nodes are named as in the imported graph.
compiler = DynamoCompiler(
    primary_registry={**tosa.ops_registry, **quant.ops_registry},
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(model_dir / "trace/trace.toml")),
)
with torch.no_grad():
    (graph,) = compiler.importer(model, *inputs)
# forward takes the inputs in the order of BertLogits.forward.
graph.order_inputs(inputs)
params = compiler.imported_params[graph]
names = {parameter.data_ptr(): name for name, parameter in model.named_parameters()}
parameter_names = {
    node.name: names[value.data_ptr()]
    for node, value in zip(graph.params, params, strict=True)
}

# ===----------------------------------------------------------------------===
# Step 2.1: Export the original graph and FP32 parameters.
# ===----------------------------------------------------------------------===
graph.dump(output / "01-import.txt")

# ===----------------------------------------------------------------------===
# Step 2.2: Absorb the transposes PyTorch puts on the Linear weights.
# ===----------------------------------------------------------------------===
permute(graph)
graph.dump(output / "02-permuted.txt")

# ===----------------------------------------------------------------------===
# Step 2.3: Quantize every Linear and attention product to INT8: the goban
# matmul thread computes in INT8 only.
# ===----------------------------------------------------------------------===
# The activation scales come from buddy-pytorch-calibrate.py.
calibration = json.loads((model_dir / "output/calibration.json").read_text())
params = quantize(graph, params, parameter_names, calibration)
graph.dump(output / "03-quanted.txt")

# ===----------------------------------------------------------------------===
# Step 3: Partition the graph: one goban hart runs all of it.
# ===----------------------------------------------------------------------===
graph.fuse_ops([simply_fuse])
graph.dump(output / "04-partitioned.txt")
driver = GraphDriver(graph)
for index, subgraph in enumerate(driver.subgraphs):
    subgraph.lower_to_top_level_ir(buffer_layout="contiguous", writable=False)
    (output / f"subgraph{index}.mlir").write_text(str(subgraph._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
# forward takes the parameters grouped by type: FP32, INT32 biases, INT8
# weights.
for index, dtype in enumerate((torch.float32, torch.int32, torch.int8)):
    np.concatenate(
        [p.detach().numpy().reshape(-1) for p in params if p.dtype == dtype]
    ).tofile(output / f"arg{index}.data")

# The runner tokenizes the prompt with the vocabulary of the checkpoint.
tokenizer = BertTokenizer.from_pretrained(bert.checkpoint)
vocabulary = tokenizer.get_vocab()
(output / "vocab.txt").write_text(
    "".join(token + "\n" for token in sorted(vocabulary, key=vocabulary.get))
)
sample = tokenizer(
    (bert.checkpoint.parent.parent / "input.txt").read_text().strip(),
    padding="max_length",
    max_length=32,
    truncation=True,
)
np.asarray(sample["input_ids"], dtype=np.int64).tofile(output / "input_ids.data")
np.asarray(sample["attention_mask"], dtype=np.int64).tofile(
    output / "attention_mask.data"
)

# ===----------------------------------------------------------------------===
# Step 5: The reference of the baremetal check: the logits of the INT8 model
# of buddy-pytorch-calibrate.py on the sample sentence. Every Linear reads
# its weight with one scale per output channel and its input with one scale;
# every attention product reads both operands with one scale each; softmax
# and GELU are those of the goban operators (exp.py).
# ===----------------------------------------------------------------------===
from torch.overrides import TorchFunctionMode

from compiler.frontend.quant import quantize as quantize_codes, quantize_bias
from models.archs.buckyball.goban.exp import gelu, softmax


class Products(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.index = 0

    def __torch_function__(self, function, types, args=(), kwargs=None):
        if getattr(function, "__name__", None) == "softmax":
            return softmax(args[0], args[1] if len(args) > 1 else kwargs["dim"])
        if getattr(function, "__name__", None) == "gelu":
            return gelu(args[0])
        if getattr(function, "__name__", None) == "matmul":
            scales = [
                torch.tensor(calibration[f"matmul.{self.index}.{side}"] / 127)
                for side in (0, 1)
            ]
            args = tuple(
                quantize_codes(operand, 1 / scale) * scale
                for operand, scale in zip(args, scales)
            )
            self.index += 1
        return function(*args, **(kwargs or {}))


reference = bert.load()
for name, module in reference.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    step = module.weight.detach().abs().amax(1, keepdim=True) / 127
    module.weight.data = quantize_codes(module.weight.detach(), 1 / step) * step
    scale = torch.tensor(calibration[name] / 127)
    bias_step = step.reshape(-1) * scale
    module.bias.data = quantize_bias(module.bias.detach(), bias_step) * bias_step
    module.register_forward_pre_hook(
        lambda module, args, scale=scale: quantize_codes(args[0], 1 / scale) * scale
    )
sentence = (
    torch.tensor([sample["input_ids"]]),
    torch.tensor([sample["token_type_ids"]]),
    torch.tensor([sample["attention_mask"]]),
    torch.arange(32)[None, :],
)
with torch.no_grad(), Products():
    logits = reference(*sentence)
logits.numpy().astype(np.float32).tofile(output / "reference.data")

# ===----------------------------------------------------------------------===
# The dataset of the accuracy runs (dataset.cpp, from the model region): the
# first 1000 validation
# sentences of emotion (the accuracy run of the design stays within two hours), tokenized as the sentence above, the logits of the
# same INT8 reference on each (the reference a sample must match) and their
# labels.
# ===----------------------------------------------------------------------===
emotion = import_module("models.models.bert-base-uncased-emotion.data")
tests, labels = emotion.load("validation", 1000)
(output / "dataset").mkdir(exist_ok=True)
tests[0].numpy().astype(np.int64).tofile(output / "dataset/ids.data")
tests[2].numpy().astype(np.int64).tofile(output / "dataset/masks.data")
with torch.no_grad(), Products():
    logits = reference(*tests)
logits.numpy().astype(np.float32).tofile(output / "dataset/reference.data")
labels.numpy().astype(np.int64).tofile(output / "dataset/labels.data")
