from pathlib import Path
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


class Loader(BaseModelLoader):
    def download_model(self, model_config):
        if not Path(model_config.model).is_dir():
            raise FileNotFoundError(model_config.model)

    def load_weights(self, model, model_config):
        model.load_device_weights()
