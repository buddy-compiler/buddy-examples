from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .execute import Thinker
from .media.embeddings import Encoders


class Model:
    def __init__(self, directory, metadata, timeout, config):
        self.config = config.thinker_config
        with ThreadPoolExecutor(max_workers=2) as pool:
            thinker = pool.submit(Thinker, directory, metadata, timeout)
            encoders = pool.submit(Encoders, directory, metadata, timeout)
            self.driver = thinker.result()
            self.media = encoders.result()

    def forward(self, tokens, positions, start, media=None, capture=()):
        embeddings, deepstack = None, None
        if media:
            embeddings = torch.from_numpy(np.frombuffer(self.driver.embeddings(tokens), dtype="<f4")
                                          .copy().reshape(-1, self.driver.hidden))
            values = self.media.encode(**media)
            ids = torch.tensor(tokens)
            mask = ((ids == self.config.audio_token_id) | (ids == self.config.image_token_id)
                    | (ids == self.config.video_token_id))
            embeddings, deepstack = self.media.merge(ids, embeddings, values, mask, self.config)
            embeddings = embeddings.numpy()
            deepstack = None if deepstack is None else deepstack.numpy()
        result = self.driver.forward(tokens, positions, start, embeddings, deepstack, capture)
        if not np.isfinite(result).all():
            raise RuntimeError("non-finite Thinker hidden states")
        return torch.from_numpy(result)

    def logits(self, hidden):
        values = self.driver.logits(hidden.detach().cpu().numpy())
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite Thinker logits")
        return torch.from_numpy(values)[0]

    def close(self):
        self.driver.close()
        self.media.close()
