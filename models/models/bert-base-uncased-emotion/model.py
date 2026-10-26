import torch


class BertLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids):
        hidden = self.model.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(
            hidden.dtype
        ).min
        for layer in self.model.bert.encoder.layer:
            hidden = layer(hidden, attention_mask=mask)
        pooled = self.model.bert.pooler(hidden)
        return self.model.classifier(self.model.dropout(pooled))
