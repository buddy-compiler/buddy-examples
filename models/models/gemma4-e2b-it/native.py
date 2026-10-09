import torch


class NativeGemma(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, past_key_values):
        keys = torch.arange(512, device=input_ids.device)[None, None, None, :]
        positions = position_ids[:, None, :, None]
        full = keys <= positions
        sliding = full & (keys > positions - 512)
        masked = torch.finfo(torch.float32).min
        return self.model(input_ids=input_ids, position_ids=position_ids,
                          past_key_values=past_key_values, use_cache=True,
                          attention_mask={"full_attention": torch.where(full, 0.0, masked),
                                          "sliding_attention": torch.where(sliding, 0.0, masked)})
