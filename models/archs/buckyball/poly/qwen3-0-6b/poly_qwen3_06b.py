def register():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("examples.models.poly.qwen3-0-6b.serve", "examples.models.poly.qwen3-0-6b.serve:Model")
