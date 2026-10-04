def register():
    from importlib import import_module
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.model_loader import register_model_loader

    module = "examples.models.omni.qwen3-omni-30b-a3b-instruct"
    ModelRegistry.register_model(module + ".model", module + ".model:Model")
    ModelRegistry.register_model(
        module + ".omni_model", module + ".omni_model:OmniModel"
    )
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    OmniModelRegistry.register_model(
        module + ".omni_model", module + ".omni_model:OmniModel"
    )
    register_model_loader("bemu_omni")(import_module(module + ".loader").Loader)
