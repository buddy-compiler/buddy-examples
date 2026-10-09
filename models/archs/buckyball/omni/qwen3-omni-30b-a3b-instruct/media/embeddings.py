from concurrent.futures import ThreadPoolExecutor

import torch

from .audio import Audio
from .vision import Vision


class Encoders:
    def __init__(self, directory, metadata, timeout):
        with ThreadPoolExecutor(max_workers=2) as pool:
            vision = pool.submit(Vision, directory, metadata, timeout)
            audio = pool.submit(Audio, directory, metadata, timeout)
            self.vision = vision.result()
            self.audio = audio.result()

    def encode(self, **kwargs):
        outputs = []
        for key, values in kwargs.items():
            if key in ("pixel_values", "pixel_values_videos"):
                grids = kwargs[
                    "image_grid_thw" if key == "pixel_values" else "video_grid_thw"
                ]
                offset = 0
                for grid in grids:
                    count = int(grid.prod())
                    result = self.vision.encode(
                        values[offset : offset + count], grid[None]
                    )
                    outputs.append(result.transpose(0, 1).reshape(result.shape[1], -1))
                    offset += count
                if offset != values.shape[0]:
                    raise ValueError("vision grids do not cover pixel values")
            elif key == "input_audio_features":
                lengths = kwargs["audio_feature_lengths"].tolist()
                if values.ndim == 3:
                    if values.shape[0] != len(lengths):
                        raise ValueError("audio batch differs from feature lengths")
                    for index, length in enumerate(lengths):
                        outputs.append(self.audio.encode(values[index, :, :length]))
                elif values.ndim == 2:
                    offset = 0
                    for length in lengths:
                        outputs.append(
                            self.audio.encode(values[:, offset : offset + length])
                        )
                        offset += length
                    if offset != values.shape[1]:
                        raise ValueError("audio lengths do not cover feature frames")
                else:
                    raise ValueError(
                        "audio features require a mel matrix or padded batch"
                    )
        return tuple(outputs)

    def merge(
        self, input_ids, embeddings, multimodal_embeddings, is_multimodal, config
    ):
        width = embeddings.shape[1]
        vision = [
            value for value in multimodal_embeddings if value.shape[-1] == 4 * width
        ]
        audio = [value for value in multimodal_embeddings if value.shape[-1] == width]
        if (
            len(vision) + len(audio) != len(multimodal_embeddings)
            or is_multimodal is None
        ):
            raise ValueError("invalid multimodal embedding contract")
        input_ids = input_ids.cpu()
        vision_mask = is_multimodal & (
            (input_ids == config.image_token_id) | (input_ids == config.video_token_id)
        )
        audio_mask = is_multimodal & (input_ids == config.audio_token_id)
        if int((vision_mask | audio_mask).sum()) != int(is_multimodal.sum()):
            raise ValueError("unrecognized multimodal placeholder token")
        deepstack = None
        if vision:
            features = torch.cat(vision)
            if int(vision_mask.sum()) != features.shape[0]:
                raise ValueError("vision embeddings do not match placeholder tokens")
            embeddings[vision_mask] = features[:, :width]
            deepstack = torch.zeros(3, embeddings.shape[0], width)
            deepstack[:, vision_mask] = (
                features[:, width:].reshape(-1, 3, width).transpose(0, 1)
            )
        elif vision_mask.any():
            raise ValueError("vision placeholder has no embedding")
        if audio:
            features = torch.cat(audio)
            if int(audio_mask.sum()) != features.shape[0]:
                raise ValueError("audio embeddings do not match placeholder tokens")
            embeddings[audio_mask] = features
        elif audio_mask.any():
            raise ValueError("audio placeholder has no embedding")
        return embeddings, deepstack

    def close(self):
        self.vision.close()
        self.audio.close()
