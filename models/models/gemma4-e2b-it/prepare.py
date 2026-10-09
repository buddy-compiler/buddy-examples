import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma4 import Gemma4ForCausalLM


def load_model(checkpoint):
    # Load the full multimodal model and extract language model weights.
    print("Loading full model from:", checkpoint)
    full_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, low_cpu_mem_usage=True
    )
    full_sd = full_model.state_dict()

    causal_sd = {}
    for k, v in full_sd.items():
        if k.startswith("model.language_model."):
            causal_sd[k.replace("model.language_model.", "model.")] = v
        elif k.startswith("lm_head."):
            causal_sd[k] = v

    if "lm_head.weight" not in causal_sd and "model.embed_tokens.weight" in causal_sd:
        causal_sd["lm_head.weight"] = causal_sd["model.embed_tokens.weight"]

    full_config = AutoConfig.from_pretrained(checkpoint)
    tc = full_config.text_config
    model = Gemma4ForCausalLM(tc)
    model.load_state_dict(causal_sd, strict=True)
    model.eval()
    del full_model, full_sd, causal_sd
    return model, tc


def export_vocabulary(checkpoint, output):
    # Export vocabulary file for the C++ tokenizer.
    print("Exporting vocabulary...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_path = output
    with open(vocab_path, "w", encoding="utf-8") as vf:
        for token, _ in sorted_vocab:
            vf.write(token.replace("\n", "\\n") + "\n")
    print(f"Exported {len(sorted_vocab)} tokens to {vocab_path}")
