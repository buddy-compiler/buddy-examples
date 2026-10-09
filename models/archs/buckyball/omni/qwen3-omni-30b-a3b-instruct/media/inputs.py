from types import SimpleNamespace

import numpy as np
import soundfile
import torch
from PIL import Image

from .positions import positions


def video_frames(path):
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened():
            raise ValueError(f"cannot open video: {path}")
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for _ in range(count):
            ok, frame = reader.read()
            if not ok:
                raise ValueError(f"incomplete video: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames)
    finally:
        reader.release()


def prepare(processor, item, base, config):
    if isinstance(item, str):
        item = {"text": item}
    if "text" not in item or set(item) - {"text", "image", "audio", "video"}:
        raise ValueError("media request requires text and known media fields")
    content, media = [], {}
    for kind, value in item.items():
        if kind == "text":
            content.append({"type": "text", "text": value})
            continue
        path = (base / value).resolve(strict=True)
        content.append({"type": kind})
        if kind == "image":
            with Image.open(path) as image:
                media["images"] = [image.convert("RGB")]
        elif kind == "video":
            media["videos"] = [video_frames(path)]
        else:
            audio, rate = soundfile.read(path, dtype="float32")
            if audio.ndim != 1:
                raise ValueError("audio input must be mono")
            target = processor.feature_extractor.sampling_rate
            if rate != target:
                from torchaudio.transforms import Resample
                threads = torch.get_num_threads()
                try:
                    torch.set_num_threads(1)
                    with torch.backends.mkldnn.flags(enabled=False):
                        audio = Resample(rate, target)(torch.from_numpy(audio)).numpy()
                finally:
                    torch.set_num_threads(threads)
            media["audio"] = [audio]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
    batch = processor(text=prompt, **media, return_tensors="pt", min_pixels=1024, max_pixels=65536,
                      sampling_rate=processor.feature_extractor.sampling_rate, use_audio_in_video=False)
    ids = batch["input_ids"][0]
    encoded = {}
    for name in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        if name in batch:
            encoded[name] = batch[name]
    if "input_features" in batch:
        encoded["input_audio_features"] = batch["input_features"]
        encoded["audio_feature_lengths"] = batch["feature_attention_mask"].sum(-1)
    features = []
    indices = {"image": 0, "video": 0, "audio": 0}
    for offset, token in enumerate(ids.tolist()):
        if token == config.audio_start_token_id:
            kind = "audio"
            data = {"audio_feature_lengths": encoded["audio_feature_lengths"][indices[kind]]}
        elif token == config.vision_start_token_id:
            kind = {config.image_token_id: "image", config.video_token_id: "video"}[int(ids[offset + 1])]
            data = {f"{kind}_grid_thw": encoded[f"{kind}_grid_thw"][indices[kind]]}
            if kind == "video":
                data["second_per_grid_ts"] = torch.as_tensor(batch["video_second_per_grid"][indices[kind]])
        else:
            continue
        indices[kind] += 1
        features.append(SimpleNamespace(modality=kind, mm_position=SimpleNamespace(offset=offset),
                                        data={key: SimpleNamespace(data=value) for key, value in data.items()}))
    pos, delta = positions(ids.tolist(), features, config)
    return ids.tolist(), pos.numpy(), delta, encoded
