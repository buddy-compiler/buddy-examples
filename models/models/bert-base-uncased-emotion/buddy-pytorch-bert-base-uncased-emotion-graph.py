from transformers import BertForSequenceClassification
from pathlib import Path
import torch
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)

checkpoint = output / "bert-base-uncased-emotion"
from model import BertLogits

model = BertLogits(
    BertForSequenceClassification.from_pretrained(
        checkpoint, attn_implementation="eager"
    )
).eval()
inputs = (
    torch.zeros((1, 32), dtype=torch.int64),
    torch.zeros((1, 32), dtype=torch.int64),
    torch.ones((1, 32), dtype=torch.int64),
    torch.arange(32, dtype=torch.int64)[None, :],
)

compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    verbose=True,
    verbose_path=output / "buddy-graph.txt",
    trace=None,
)
with torch.no_grad():
    graphs = compiler.importer(model, *inputs)
for graph in graphs:
    graph.lower_to_top_level_ir()
