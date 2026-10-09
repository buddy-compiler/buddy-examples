from contextlib import ExitStack

import numpy as np
import torch

from .conditioning import prefill
from .model import Model
from .speech_model import Speech, Vocoder


def sample(logits, temperature):
    if logits.ndim != 1 or not torch.isfinite(logits).any() or torch.isnan(logits).any():
        raise ValueError("invalid sampling logits")
    if temperature == 0:
        return int(logits.argmax())
    return int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))


class OmniModel:
    def __init__(self, directory, metadata, timeout, config):
        self.config = config
        self.resources = ExitStack()
        try:
            self.thinker = Model(directory, metadata, timeout, config)
            self.resources.callback(self.thinker.close)
            self.talker = Speech(directory, metadata, timeout, config.talker_config)
            self.resources.callback(self.talker.driver.close)
            self.wave = Vocoder(directory, metadata, timeout)
            self.resources.callback(self.wave.driver.close)
        except BaseException:
            self.resources.close()
            raise

    def generate(self, tokens, positions, delta, media, max_tokens, max_audio_tokens, temperature, eos):
        if max_tokens < 1 or max_audio_tokens < 2 or not np.isfinite(temperature) or temperature < 0:
            raise ValueError("invalid generation limits or temperature")
        if len(tokens) + max_tokens - 1 > self.thinker.driver.capacity:
            raise ValueError("request exceeds compiled Thinker cache capacity")
        layer = self.config.talker_config.accept_hidden_layer
        captures = {0: [], layer: []}
        generated, start, request = [], 0, tokens
        for step in range(max_tokens):
            hidden = self.thinker.forward(request, positions, start, media if start == 0 else None,
                                          capture=(0, layer))
            for index in captures:
                captures[index].append(torch.from_numpy(self.thinker.driver.captures[index].copy()))
            token = sample(self.thinker.logits(hidden[-1:]), temperature)
            generated.append(token)
            start += len(request)
            if token in eos:
                break
            request = [token]
            positions = np.full((3, 1), start + delta, dtype="<i8")
        executed = torch.tensor(tokens + generated[:-1])
        embeds, accepted = (torch.cat(captures[index]) for index in (0, layer))
        special_ids = [self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]
        special = torch.from_numpy(np.frombuffer(self.thinker.driver.embeddings(special_ids), dtype="<f4")
                                   .copy().reshape(3, -1))
        conditioning, trailing, pad = prefill(self.config, self.talker, torch.tensor(tokens), executed,
                                              embeds, accepted, special)
        count = len(conditioning)
        if count + max_audio_tokens - 1 > self.talker.driver.capacity:
            raise ValueError("request exceeds compiled Talker cache capacity")
        if max_audio_tokens - 1 > self.wave.driver.capacity:
            raise ValueError("audio limit exceeds compiled Wave capacity")
        hidden = self.talker.driver.forward(conditioning, torch.arange(count).repeat(3, 1), 0)
        codes = []
        for step in range(max_audio_tokens):
            logits = self.talker.compute_logits(hidden[-1:])[0].clone()
            eos_id = self.config.talker_config.codec_eos_token_id
            eos_logit = logits[eos_id].clone()
            logits[-1024:] = -torch.inf
            logits[eos_id] = eos_logit
            token = sample(logits, 0)
            if token == eos_id or step == max_audio_tokens - 1:
                break
            frame, summed = self.talker.driver.codes(torch.tensor([token]), hidden[-1:])
            codes.append(frame[0, :, 0])
            text = trailing[:1]
            trailing = trailing[1:] if len(trailing) > 1 else pad
            inputs = summed.reshape(1, -1) + text
            hidden = self.talker.driver.forward(inputs, torch.full((3, 1), count), count)
            count += 1
        if not codes:
            raise ValueError("Talker finished without a complete acoustic frame")
        frames = torch.stack(codes, dim=1)
        if (frames < 0).any() or (frames >= self.config.code2wav_config.codebook_size).any():
            raise ValueError("Talker emitted an invalid acoustic code")
        return generated, self.wave.driver.decode(frames)

    def close(self):
        self.resources.close()
