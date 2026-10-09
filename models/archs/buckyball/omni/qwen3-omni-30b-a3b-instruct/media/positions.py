import numpy as np
import torch


def positions(input_tokens, features, config):
    total = len(input_tokens)
    chunks = []
    consumed = 0
    videos = [
        f
        for f in features
        if f.modality == "video"
        and f.data.get("use_audio_in_video")
        and f.data["use_audio_in_video"].data.item()
    ]
    audios = [f for f in features if f.modality == "audio"]
    pairs = dict(
        zip((f.mm_position.offset for f in videos), audios[: len(videos)], strict=True)
    )
    paired = {f.mm_position.offset for f in pairs.values()}
    for feature in sorted(features, key=lambda f: f.mm_position.offset):
        offset = feature.mm_position.offset
        if feature.modality == "audio" and offset in paired:
            continue
        origin = int(chunks[-1].max()) + 1 if chunks else 0
        if offset > consumed:
            chunks.append(
                np.broadcast_to(np.arange(offset - consumed), (3, offset - consumed))
                + origin
            )
            origin += offset - consumed
        chunks.append(np.full((3, 1), origin, dtype=np.int64))
        origin += 1
        if feature.modality == "audio":
            frames = feature.data["audio_feature_lengths"].data.item()
            count = frames // 100 * 13 + (frames % 100 + 7) // 8
            chunks.append(np.broadcast_to(np.arange(count), (3, count)) + origin)
            chunks.append(np.full((3, 1), origin + count, dtype=np.int64))
            consumed = offset + count + 2
            continue
        if feature.modality not in ("image", "video"):
            raise ValueError(f"unknown modality: {feature.modality}")
        key = "image_grid_thw" if feature.modality == "image" else "video_grid_thw"
        temporal, height, width = feature.data[key].data.tolist()
        merge = config.vision_config.spatial_merge_size
        height //= merge
        width //= merge
        indices = np.indices((temporal, height, width)).reshape(3, -1)
        factor = config.position_id_per_seconds
        if feature.modality == "video":
            factor *= feature.data["second_per_grid_ts"].data.item()
        indices[0] = (indices[0] * factor).astype(np.int64)
        indices += origin
        if offset in pairs:
            chunks.append(np.full((3, 1), origin - 1, dtype=np.int64))
            frames = pairs[offset].data["audio_feature_lengths"].data.item()
            count = frames // 100 * 13 + (frames % 100 + 7) // 8
            audio = np.broadcast_to(np.arange(count), (3, count)) + origin
            combined = np.concatenate((indices, audio), axis=1)
            # Stable ordering puts video tokens first when timestamps are equal.
            combined = combined[:, np.argsort(combined[0], kind="stable")]
            chunks.append(combined)
            end = np.full((3, 1), int(combined.max()) + 1, dtype=np.int64)
            chunks.extend((end, end))
            consumed = offset + 4 + temporal * height * width + count
        else:
            chunks.append(indices)
            chunks.append(np.full((3, 1), int(indices.max()) + 1, dtype=np.int64))
            consumed = offset + 2 + temporal * height * width
    if consumed < total:
        origin = int(chunks[-1].max()) + 1 if chunks else 0
        chunks.append(
            np.broadcast_to(np.arange(total - consumed), (3, total - consumed)) + origin
        )
    result = np.concatenate(chunks, axis=1)
    if result.shape != (3, total):
        raise ValueError("multimodal position length differs from token sequence")
    return torch.from_numpy(result.copy()), int(result.max()) + 1 - total
