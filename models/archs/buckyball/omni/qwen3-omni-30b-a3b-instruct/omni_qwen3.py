def register():
    from importlib import import_module
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.model_loader import register_model_loader

    module = "examples.models.omni.qwen3-omni-30b-a3b-instruct"
    ModelRegistry.register_model(module + ".model", module + ".model:Model")
    register_model_loader("bemu_omni")(import_module(module + ".loader").Loader)
