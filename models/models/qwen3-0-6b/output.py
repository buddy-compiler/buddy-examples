from torch import nn

class OutputShard(nn.Module):
    def __init__(self, model, rank, tiles):
        super().__init__()
        vocabulary, hidden = model.lm_head.weight.shape
        if vocabulary % tiles:
            raise ValueError("output tiles must divide the vocabulary")
        rows = vocabulary // tiles
        self.norm = model.model.norm
        self.head = nn.Linear(hidden, rows, bias=False, device="meta")
        self.head.weight = nn.Parameter(
            model.lm_head.weight[rank * rows : (rank + 1) * rows].detach(),
            requires_grad=False,
        )

    def forward(self, hidden):
        return self.head(self.norm(hidden))
