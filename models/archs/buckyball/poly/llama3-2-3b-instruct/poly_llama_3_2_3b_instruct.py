def register():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("examples.models.poly.llama3-2-3b-instruct.serve", "examples.models.poly.llama3-2-3b-instruct.serve:Model")
