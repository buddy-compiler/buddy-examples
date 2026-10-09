import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def prepare(checkpoint):
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
    return model, PreTrainedTokenizerFast.from_pretrained(checkpoint)
