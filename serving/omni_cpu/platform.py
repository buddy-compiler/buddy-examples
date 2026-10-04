from vllm.platforms.cpu import CpuPlatform
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum


class Platform(OmniPlatform, CpuPlatform):
    _omni_enum = OmniPlatformEnum.OOT

    @classmethod
    def get_omni_ar_worker_cls(cls):
        return "stack.serving.omni_cpu.worker.AutoregressiveWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls):
        return "stack.serving.omni_cpu.worker.GenerationWorker"

    @classmethod
    def get_device_count(cls):
        return 1
