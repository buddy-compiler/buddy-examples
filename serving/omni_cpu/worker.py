import torch
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.cpu_worker import CPUWorker
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class AutoregressiveRunner(GPUARModelRunner, CPUModelRunner):
    pass


class GenerationRunner(GPUGenerationModelRunner, CPUModelRunner):
    pass


class AutoregressiveWorker(OmniWorkerMixin, CPUWorker):
    model_runner_cls = AutoregressiveRunner

    def init_device(self):
        super().init_device()
        self.vllm_config.compilation_config.max_cudagraph_capture_size = 0
        self.model_runner = AutoregressiveRunner(self.vllm_config, torch.device("cpu"))


class GenerationWorker(OmniWorkerMixin, CPUWorker):
    model_runner_cls = GenerationRunner

    def init_device(self):
        super().init_device()
        self.vllm_config.compilation_config.max_cudagraph_capture_size = 0
        self.model_runner = GenerationRunner(self.vllm_config, torch.device("cpu"))
