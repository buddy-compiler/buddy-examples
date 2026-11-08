import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma4 import Gemma4ForCausalLM
from pathlib import Path
from huggingface_hub import snapshot_download

output = Path(__file__).resolve().parent / "output"
output.mkdir(exist_ok=True)
checkpoint = snapshot_download(
    "google/gemma-4-E2B-it",
)
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
prepared = output / "gemma4-e2b-it"
model.save_pretrained(prepared)
AutoTokenizer.from_pretrained(checkpoint).save_pretrained(prepared)

snapshot_download(
    "Salesforce/wikitext",
    repo_type="dataset",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    allow_patterns=["wikitext-2-raw-v1/*.parquet"],
    local_dir=Path(__file__).resolve().parents[2] / "datasets/wikitext",
)
