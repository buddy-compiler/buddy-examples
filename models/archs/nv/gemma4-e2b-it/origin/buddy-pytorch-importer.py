from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.models.gemma4 import Gemma4ForCausalLM
from torch._inductor.decomposition import decompositions

root = Path(__file__).resolve().parents[5]
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from stack.compiler.frontend.ops import tosa
from buddy.compiler.trace import TraceConfig, load_trace_config

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)
checkpoint = (
    Path(__file__).resolve().parent
    / "../../../../models/gemma4-e2b-it/output/gemma4-e2b-it"
).resolve()
model = Gemma4ForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
model.config.use_cache = False


# Recompute the visible context on each token; no hidden KV-cache state is assumed.
class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, mask):
        return self.model(input_ids=tokens, attention_mask=mask, use_cache=False).logits


compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=decompositions,
    trace=TraceConfig(load_trace_config(Path(__file__).parent / "trace/trace.toml")),
)
with torch.no_grad():
    graphs = compiler.importer(
        Logits(model),
        torch.zeros((1, 32), dtype=torch.int64),
        torch.ones((1, 32), dtype=torch.int64),
    )
assert len(graphs) == 1
graph = graphs[0]
params = compiler.imported_params[graph]
graph.fuse_ops([simply_fuse])
driver = GraphDriver(graph)
driver.subgraphs[0].lower_to_top_level_ir()
(output / "subgraph0.mlir").write_text(str(driver.subgraphs[0]._imported_module))
(output / "forward.mlir").write_text(str(driver.construct_main_graph(True)))
np.concatenate([p.detach().cpu().float().numpy().reshape(-1) for p in params]).tofile(
    output / "arg0.data"
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
with (output / "vocab.txt").open("w") as stream:
    for token, index in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
        stream.write(token.replace("\n", "\\n") + "\n")
