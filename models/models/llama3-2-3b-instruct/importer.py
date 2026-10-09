import json
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch


def prepare_checkpoint(model_id, directory):
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
    tokenizer.save_pretrained(directory / "tokenizer")
    model.config.save_pretrained(directory / "tokenizer")
    model.generation_config.save_pretrained(directory / "tokenizer")
    path = directory / "tokenizer/tokenizer_config.json"
    config = json.loads(path.read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    path.write_text(json.dumps(config, indent=2))
    return model, tokenizer
