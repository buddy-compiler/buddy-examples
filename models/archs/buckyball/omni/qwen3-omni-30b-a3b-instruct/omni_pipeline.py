from dataclasses import replace

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE


def register():
    from .omni_model import OmniModel
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(f"{__package__}.omni_model", OmniModel)
    pipeline = replace(
        QWEN3_OMNI_PIPELINE,
        model_arch=f"{__package__}.omni_model",
        duplex_plugin=None,
        stages=(
            replace(
                QWEN3_OMNI_PIPELINE.stages[0],
                custom_process_next_stage_input_func=f"{__package__}.conditioning.thinker_payload",
            ),
            replace(
                QWEN3_OMNI_PIPELINE.stages[1],
                custom_process_next_stage_input_func=f"{__package__}.conditioning.codec_payload",
            ),
            QWEN3_OMNI_PIPELINE.stages[2],
        ),
    )
    register_pipeline(pipeline)
