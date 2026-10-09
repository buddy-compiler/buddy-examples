import torch


def export_phase(exporter, phase, causal_model, parameters, tiles, tokens, design):
    model = causal_model.model
    first_event = len(exporter.calls)
    hidden = model.config.hidden_size
    layers = model.config.num_hidden_layers
    width = model.hidden_size_per_layer_input
    prefill = phase == "prefill"
    layer_zero = lambda *shape: torch.zeros(shape)
    slot = lambda name, **fields: {"slot": name, **fields}

    def matrix(projection):
        packed, layout = parameters.weight(projection.weight, matrix=True)
        span = design.partition.panel_ranges(
            projection.out_features, projection.in_features, layout, [tiles[0]]
        )[0]
        return design.linear.PackedLinear(packed, projection.in_features, layout, span)

    def emit(
        stage,
        task,
        inputs,
        names,
        outputs,
        layer=-1,
        rank=-1,
        target="ffn",
        panels=None,
    ):
        exporter.add(
            phase,
            layer,
            rank,
            stage,
            task,
            inputs,
            names,
            parameters,
            target=target,
            outputs=outputs,
            panels=panels,
        )

    ids = torch.zeros((1, tokens), dtype=torch.int64)
    positions = torch.arange(tokens, dtype=torch.int64).reshape(1, tokens)
    for name, embedding, output in (
        ("embedding", model.embed_tokens, "hidden.0"),
        ("per_layer_embedding", model.embed_tokens_per_layer, "per_layer_embedding"),
    ):
        packed, _ = parameters.weight(embedding.weight, matrix=False)
        emit(
            name,
            design.entry.Embedding(embedding, packed),
            [ids],
            [slot("tokens")],
            [output],
        )
    emit(
        "per_layer_inputs",
        design.entry.PerLayerInputs(model, matrix(model.per_layer_model_projection)),
        [layer_zero(1, tokens, hidden), layer_zero(1, tokens, layers * width)],
        [slot("hidden.0"), slot("per_layer_embedding")],
        ["per_layer_inputs"],
    )
    for kind in sorted(set(model.config.layer_types)):
        emit(
            "positions." + kind,
            design.entry.Positions(model, kind),
            [layer_zero(1, tokens, hidden), positions],
            [slot("hidden.0"), slot("positions")],
            [kind + ".cosine", kind + ".sine", kind + ".mask"],
            target="attention",
        )

    for index, layer in enumerate(model.layers):
        projections = dict(
            q=layer.self_attn.q_proj,
            o=layer.self_attn.o_proj,
            gate=layer.mlp.gate_proj,
            up=layer.mlp.up_proj,
            down=layer.mlp.down_proj,
            input_gate=layer.per_layer_input_gate,
            input_projection=layer.per_layer_projection,
        )
        if not layer.self_attn.is_kv_shared_layer:
            projections.update(k=layer.self_attn.k_proj, v=layer.self_attn.v_proj)
        weights, layouts = {}, {}
        for name, projection in projections.items():
            weights[name], layouts[name] = parameters.weight(
                projection.weight, matrix=True
            )
        bundle = design.tasks.layer_tasks(
            layer, weights, layouts, tiles, prefill=prefill
        )
        kind = model.config.layer_types[index]
        depth = layer.self_attn.head_dim
        prefix = f"layer{index}."
        exporter.calls.append(
            {
                "kind": "view",
                "phase": phase,
                "layer": index,
                "input": "per_layer_inputs",
                "output": prefix + "per_layer_input",
                "axis": 2,
                "start": index * width,
                "count": width,
            }
        )
        cos, sin = model.rotary_emb(layer_zero(1, tokens, hidden), positions, kind)
        mask = torch.zeros((1, 1, tokens, 512))
        cache = bundle["kv"]["source"]
        cache_slots = [
            f"cache{cache}.{name}"
            for name in ("key_codes", "key_scales", "value_codes", "value_scales")
        ]
        emit(
            "prepare",
            bundle["prepare"],
            [layer_zero(1, tokens, hidden)],
            [slot(f"hidden.{index}")],
            [prefix + "normalized"],
            index,
        )
        if "kv_producer" in bundle:
            deltas = [
                prefix + name + "_delta"
                for name in ("key_codes", "key_scales", "value_codes", "value_scales")
            ]
            emit(
                "kv",
                bundle["kv_producer"],
                [layer_zero(1, tokens, hidden), cos, sin],
                [
                    slot(prefix + "normalized"),
                    slot(kind + ".cosine"),
                    slot(kind + ".sine"),
                ],
                deltas,
                index,
                rank=tiles.index(bundle["kv"]["owner"]),
            )
            exporter.calls.append(
                {
                    "kind": "cache_update",
                    "phase": phase,
                    "layer": index,
                    "source": cache,
                    "owner": bundle["kv"]["owner"],
                    "start": f"cache{cache}.length",
                    "count": "valid_tokens",
                    "inputs": deltas,
                    "outputs": cache_slots,
                    "storage": "mxfp8",
                }
            )
        for stage in ("q", "attention", "o", "gateup", "down"):
            if stage == "gateup":
                emit(
                    "post_attention",
                    bundle["post_attention"],
                    [layer_zero(1, tokens, hidden), layer_zero(1, tokens, hidden)],
                    [slot(prefix + "attention_projected"), slot(f"hidden.{index}")],
                    [prefix + "attention_hidden", prefix + "ffn_normalized"],
                    index,
                )
            parts, ranges = [], []
            heads = design.partition.head_ranges(
                model.config.num_attention_heads, tiles
            )
            for rank, actor in enumerate(bundle["ranks"]):
                if stage not in actor:
                    continue
                geometry = actor["ranges"].get("gate" if stage == "gateup" else stage)
                task = actor[stage]
                target = "ffn"
                if stage == "attention":
                    first, end = heads[rank]
                    geometry = {
                        "column_start": first * depth,
                        "columns": (end - first) * depth,
                    }
                    task = design.tasks.AttentionHeads(
                        layer, 0, end - first, prefill=prefill
                    )
                    inputs = [
                        layer_zero(1, tokens, geometry["columns"]),
                        cos,
                        sin,
                        torch.zeros((1, 1, 512, depth), dtype=torch.int8),
                        torch.full((1, 1, 512, depth // 32), 127, dtype=torch.int8),
                        torch.zeros((1, 1, 512, depth), dtype=torch.int8),
                        torch.full((1, 1, 512, depth // 32), 127, dtype=torch.int8),
                        mask,
                    ]
                    query_slot = prefix + f"query_heads.rank{rank}"
                    exporter.calls.append(
                        {
                            "kind": "view",
                            "phase": phase,
                            "layer": index,
                            "input": prefix + "query",
                            "output": query_slot,
                            "axis": 2,
                            "start": geometry["column_start"],
                            "count": geometry["columns"],
                        }
                    )
                    names = [
                        slot(query_slot),
                        slot(kind + ".cosine"),
                        slot(kind + ".sine"),
                        *[slot(name) for name in cache_slots],
                        slot(kind + ".mask"),
                    ]
                    target = "attention"
                else:
                    size, name = {
                        "q": (hidden, "normalized"),
                        "o": (model.config.num_attention_heads * depth, "context"),
                        "gateup": (hidden, "ffn_normalized"),
                        "down": (layer.mlp.intermediate_size, "activation"),
                    }[stage]
                    inputs, names = [layer_zero(1, tokens, size)], [slot(prefix + name)]
                    geometry = dict(
                        geometry, layout=layouts["gate" if stage == "gateup" else stage]
                    )
                output = prefix + stage + f".rank{rank}"
                emit(
                    stage, task, inputs, names, [output], index, rank, target, geometry
                )
                parts.append(output)
                ranges.append(
                    {key: geometry[key] for key in ("column_start", "columns")}
                )
            name = {
                "q": "query",
                "attention": "context",
                "o": "attention_projected",
                "gateup": "activation",
                "down": "ffn_projected",
            }[stage]
            exporter.gather(
                phase, index, stage + "_gather", parts, prefix + name, ranges
            )
        emit(
            "post_layer",
            bundle["post_layer"],
            [
                layer_zero(1, tokens, hidden),
                layer_zero(1, tokens, hidden),
                layer_zero(1, tokens, width),
            ],
            [
                slot(prefix + "ffn_projected"),
                slot(prefix + "attention_hidden"),
                slot(prefix + "per_layer_input"),
            ],
            [f"hidden.{index + 1}"],
            index,
        )

    exporter.calls.append(
        {
            "kind": "last_valid_row",
            "phase": phase,
            "layer": layers,
            "input": f"hidden.{layers}",
            "output": "last_hidden",
            "valid": "valid_tokens",
        }
    )
    emit(
        "final_norm",
        design.entry.FinalNorm(model),
        [layer_zero(1, 1, hidden)],
        [slot("last_hidden")],
        ["final_hidden"],
        layer=layers,
    )
    weight, layout = parameters.weight(causal_model.lm_head.weight, matrix=True)
    parts, ranges = [], []
    for rank, geometry in enumerate(
        design.partition.panel_ranges(model.config.vocab_size, hidden, layout, tiles)
    ):
        if not geometry["columns"]:
            continue
        projection = design.linear.PackedLinear(weight, hidden, layout, geometry)
        output = f"logits.rank{rank}"
        emit(
            "logits",
            design.entry.Logits(causal_model, projection),
            [layer_zero(1, 1, hidden)],
            [slot("final_hidden")],
            [output],
            layer=layers,
            rank=rank,
            panels=dict(geometry, layout=layout),
        )
        parts.append(output)
        ranges.append({key: geometry[key] for key in ("column_start", "columns")})
    exporter.gather(phase, layers, "logits_gather", parts, "logits", ranges)

    def qualify(name):
        return (
            name
            if name.startswith("cache") or name == "valid_tokens"
            else phase + "." + name
        )

    for event in exporter.calls[first_event:]:
        if event["kind"] == "compute":
            for binding in event["bindings"]:
                if "slot" in binding:
                    binding["slot"] = qualify(binding["slot"])
        for key in ("inputs", "outputs"):
            if key in event:
                event[key] = [qualify(name) for name in event[key]]
        for key in ("input", "output", "start", "count", "valid"):
            if key in event and isinstance(event[key], str):
                event[key] = qualify(event[key])
