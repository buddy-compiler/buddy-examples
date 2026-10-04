import torch

_FULL_PAYLOAD_REPLACE_KEYS = frozenset(
    {"embed.tts_bos", "embed.tts_eos", "embed.tts_pad"}
)


def thinker_payload(transfer_manager, pooling_output, request):
    config = transfer_manager._get_model_config().hf_config
    embeddings = pooling_output["hidden_states.layer_0"].detach().cpu()
    hidden = (
        pooling_output[
            f"hidden_states.layer_{config.talker_config.accept_hidden_layer}"
        ]
        .detach()
        .cpu()
    )
    tokens = list(request.prompt_token_ids) + list(request.output_token_ids)
    if embeddings.shape != hidden.shape or embeddings.shape[0] != len(tokens) - 1:
        raise ValueError("Thinker conditioning does not match executed token positions")
    return {
        "embed": {
            "prefill": embeddings,
            **{
                name: pooling_output[f"embed.{name}"].detach().cpu()
                for name in ("tts_bos", "tts_eos", "tts_pad")
            },
        },
        "hidden_states": {"output": hidden},
        "ids": {"all": tokens[:-1], "prompt": list(request.prompt_token_ids)},
        "meta": {"finished": torch.tensor(True)},
    }


def codec_payload(transfer_manager, pooling_output, request):
    config = transfer_manager._get_model_config().hf_config
    tokens = list(request.output_token_ids)[:-1]
    if not tokens:
        raise ValueError("Talker finished without a complete acoustic frame")
    codes = pooling_output["codes.audio"][-len(tokens) :].detach().cpu()
    if codes.shape != (len(tokens), config.talker_config.num_code_groups):
        raise ValueError(
            "Talker acoustic frame shape differs from its generated tokens"
        )
    if not torch.equal(codes[:, 0], torch.tensor(tokens)):
        raise ValueError("Talker acoustic frames are not aligned with executed tokens")
    if (codes < 0).any() or (codes >= config.code2wav_config.codebook_size).any():
        raise ValueError("Talker emitted an invalid acoustic code")
    return {
        "codes": {"audio": codes.transpose(0, 1).contiguous().reshape(-1).tolist()},
        "meta": {"finished": torch.tensor(True)},
    }


class Conditioning:
    def talker_preprocess_prefill(self, input_ids, input_embeds, payload):
        embed = payload["embed"]
        ids = payload["ids"]
        start = payload["_omni_num_computed_tokens"]
        speaker = self._get_text_spk_token_id(self.default_tts_text_spk_type)
        tokens, hidden, trailing = self._thinker_to_talker_prefill(
            thinker_embed=embed["prefill"],
            thinker_hidden=payload["hidden_states"]["output"],
            multimodal_mask=None,
            input_ids=torch.tensor(ids["prompt"])[None],
            thinker_result_ids=torch.tensor(ids["all"]),
            speaker_id=speaker,
            tts_bos_thinker=embed["tts_bos"],
            tts_eos_thinker=embed["tts_eos"],
            tts_pad_thinker=embed["tts_pad"],
        )
        end = start + input_ids.shape[0]
        if end > hidden.shape[0]:
            raise ValueError("Talker scheduled span exceeds conditioning length")
        return (
            tokens[start:end],
            hidden[start:end],
            {
                "hidden_states": {"trailing_text": trailing.detach()},
                "embed": {"tts_pad_projected": self.tts_pad_embed.detach()},
                "meta": {"prefill_consumed_text_tokens": 1},
            },
        )

    def _get_tts_embed(self, thinker_embed, bos, eos, pad):
        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = (
            self.talker.text_projection(value.reshape(1, -1))
            for value in (bos, eos, pad)
        )
        return self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed

    def _get_talker_user_parts(self, start, end, mask, hidden, embeddings):
        selected = mask[start:end]
        result = torch.empty(end - start, self.talker.driver.width)
        if selected.any():
            result[selected] = self.talker.hidden_projection(
                hidden[start:end][selected]
            )
        result[~selected] = self.talker.text_projection(
            embeddings[start:end][~selected]
        )
        return result

    def _get_talker_assistant_parts(
        self, start, end, speaker, embeddings, pad, bos, eos
    ):
        hidden = self.talker.text_projection(embeddings[start:end])
        if hidden.shape[0] < 4:
            raise ValueError(
                "Talker requires assistant header and a generated text token"
            )
        text = torch.cat((hidden[:3], pad.expand(4, -1), bos, hidden[3:4]))
        config = self.talker_config
        codes = torch.tensor(
            [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
                speaker,
                config.codec_pad_id,
                config.codec_bos_id,
            ]
        )
        codec = torch.cat((torch.zeros_like(hidden[:3]), self.talker.embedding(codes)))
        trailing = torch.cat((hidden[4:], eos))
        tokens = torch.full((text.shape[0],), self.config.tts_pad_token_id)
        return text + codec, tokens, trailing

    def talker_preprocess_decode(self, input_ids, input_embeds, update, payload):
        states = payload["hidden_states"]
        trailing = states["trailing_text"]
        text = trailing[:1]
        update["hidden_states"] = {
            "trailing_text": (
                trailing[1:] if trailing.shape[0] > 1 else self.tts_pad_embed
            )
        }
        return states["last"].reshape(1, -1), text, update
