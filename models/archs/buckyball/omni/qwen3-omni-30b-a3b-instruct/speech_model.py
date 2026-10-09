from torch import nn

from .media.talker import Talker
from .media.wave import Wave


class Embedding(nn.Module):
    def __init__(self, driver):
        super().__init__()
        self.driver = driver

    def forward(self, tokens):
        return self.driver.embedding(tokens.detach().cpu().numpy()).reshape(
            *tokens.shape, self.driver.width
        )


class Projection(nn.Module):
    def __init__(self, driver, kind):
        super().__init__()
        self.driver = driver
        self.kind = kind

    def forward(self, hidden):
        return self.driver.project(hidden, self.kind)


class Speech(nn.Module):
    def __init__(self, directory, metadata, timeout, config):
        super().__init__()
        self.driver = Talker(directory, metadata, timeout)
        self.config = config.text_config
        self.num_code_groups = config.num_code_groups
        self.embedding = Embedding(self.driver)
        self.text_projection = Projection(self.driver, 0)
        self.hidden_projection = Projection(self.driver, 1)
        self.next_position = 0

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
    ):
        if multimodal_embeddings:
            raise ValueError("Talker receives codec IDs and projected embeddings")
        return self.embedding(input_ids)

    def forward(self, input_ids, positions, inputs_embeds):
        if positions.ndim == 1:
            positions = positions[None].repeat(3, 1)
        if int(positions[0, 0]) == 0:
            self.next_position = 0
        result = self.driver.forward(inputs_embeds, positions, self.next_position)
        self.next_position += result.shape[0]
        return result

    def compute_logits(self, hidden):
        return self.driver.logits(hidden)

    def code_predictor_forward(self, input_ids, input_embeds, last_talker_hidden):
        return self.driver.codes(
            input_ids, last_talker_hidden.reshape(1, self.driver.width)
        )


class Vocoder(nn.Module):
    def __init__(self, directory, metadata, timeout):
        super().__init__()
        self.driver = Wave(directory, metadata, timeout)

    def chunked_decode(
        self, codes, chunk_size, left_context_size, seq_token_counts=None
    ):
        if codes.shape[0] != 1:
            raise ValueError("Code2Wav executes one request")
        if seq_token_counts is not None and len(seq_token_counts) != 1:
            raise ValueError("Code2Wav executes one request")
        return [self.driver.decode(codes[0])]
