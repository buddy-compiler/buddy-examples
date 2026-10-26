from pathlib import Path
import torch
from diffusers import DiffusionPipeline
from buddy.compiler.frontend import DynamoCompiler
from stack.compiler.frontend.ops import tosa
from torch._inductor.decomposition import decompositions

output = Path(__file__).resolve().parent / "output"
checkpoint = output / "stable-diffusion"
pipe = DiffusionPipeline.from_pretrained(checkpoint, torch_dtype=torch.float32)
pipe.text_encoder.eval()
pipe.unet.eval()
pipe.vae.eval()


class Text(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pipe.text_encoder

    def forward(self, tokens):
        result = self.model(tokens)
        return result.last_hidden_state, result.pooler_output


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pipe.unet

    def forward(self, sample, timestep, states):
        return self.model(sample, timestep, encoder_hidden_states=states).sample


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pipe.vae

    def forward(self, sample):
        return self.model.decode(sample).sample


for name, model, inputs in (
    ("text_encoder", Text(), (torch.zeros((1, 77), dtype=torch.int64),)),
    (
        "unet",
        UNet(),
        (torch.zeros((2, 4, 64, 64)), torch.ones((1,)), torch.zeros((2, 77, 1024))),
    ),
    ("vae", VAE(), (torch.zeros((1, 4, 64, 64)),)),
):
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=decompositions,
        verbose=True,
        verbose_path=output / f"buddy-graph-{name}.txt",
        trace=None,
    )
    with torch.no_grad():
        graphs = compiler.importer(model.eval(), *inputs)
    for graph in graphs:
        graph.lower_to_top_level_ir()
(output / "buddy-graph.txt").write_text(
    "\n".join(
        (output / f"buddy-graph-{name}.txt").read_text()
        for name in ("text_encoder", "unet", "vae")
    )
)
