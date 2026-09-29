def sample_tokens(tokenizer):
    return tokenizer("The capital of France is", return_tensors="pt")["input_ids"]
