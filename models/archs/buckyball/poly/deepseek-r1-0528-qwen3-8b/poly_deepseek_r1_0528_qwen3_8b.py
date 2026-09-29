def register():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("examples.models.poly.deepseek-r1-0528-qwen3-8b.serve", "examples.models.poly.deepseek-r1-0528-qwen3-8b.serve:Model")
