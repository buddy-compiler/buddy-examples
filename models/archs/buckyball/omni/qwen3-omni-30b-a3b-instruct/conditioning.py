import torch


def prefill(config, talker, prompt_ids, executed_ids, embeddings, hidden, special):
    if embeddings.shape != hidden.shape or len(executed_ids) != len(embeddings):
        raise ValueError("Thinker conditioning does not match executed positions")
    bos, eos, pad = [talker.text_projection(value.reshape(1, -1)) for value in special]
    thinker = config.thinker_config
    mask = ((executed_ids == thinker.audio_token_id) | (executed_ids == thinker.image_token_id)
            | (executed_ids == thinker.video_token_id))
    starts = torch.nonzero(prompt_ids == config.im_start_token_id).flatten().tolist()
    starts.append(len(executed_ids))
    parts = []
    trailing = None
    speaker = next(iter(config.talker_config.speaker_id.values()))
    for index, (start, end) in enumerate(zip(starts, starts[1:])):
        role = int(prompt_ids[start + 1])
        if role == config.system_token_id:
            continue
        if role == config.user_token_id:
            selected = mask[start:end]
            values = torch.empty(end - start, talker.driver.width)
            if selected.any():
                values[selected] = talker.hidden_projection(hidden[start:end][selected])
            values[~selected] = talker.text_projection(embeddings[start:end][~selected])
            parts.append(values)
        elif role == config.assistant_token_id:
            if index != len(starts) - 2:
                continue
            values = talker.text_projection(embeddings[start:end])
            if len(values) < 4:
                raise ValueError("Talker requires assistant header and a generated text token")
            text = torch.cat((values[:3], pad.expand(4, -1), bos, values[3:4]))
            c = config.talker_config
            ids = torch.tensor([c.codec_nothink_id, c.codec_think_bos_id, c.codec_think_eos_id,
                                speaker, c.codec_pad_id, c.codec_bos_id])
            codec = torch.cat((torch.zeros_like(values[:3]), talker.embedding(ids)))
            parts.append(text + codec)
            trailing = torch.cat((values[4:], eos))
        else:
            raise ValueError("unknown ChatML role in Talker conditioning")
    if trailing is None:
        raise ValueError("Talker conditioning requires a final assistant segment")
    return torch.cat(parts), trailing, pad
