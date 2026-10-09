import math


def participants(tiles):
    tiles = tuple(tiles)
    if (
        not tiles
        or len(tiles) & (len(tiles) - 1)
        or len(set(tiles)) != len(tiles)
        or any(type(tile) is not int or tile <= 0 for tile in tiles)
    ):
        raise ValueError(
            "Gemma participants must be distinct compute tiles of power-of-two count"
        )
    return tiles


def panel_ranges(columns, reduction, layout, tiles):
    tiles = participants(tiles)
    width, chunk, stride = (layout["tile_n"], layout["tile_k"], layout["panel_stride"])
    if (
        columns <= 0
        or reduction <= 0
        or reduction % 32
        or width <= 0
        or width % 16
        or chunk <= 0
        or chunk % 32
        or layout["tile_m"] <= 0
        or (layout["tile_m"] != 1 and layout["tile_m"] % 16)
        or stride != layout["bank_bytes"]
    ):
        raise ValueError("Gemma panel partition requires the original MXFP8 layout")
    count = math.ceil(columns / width)
    panel_bytes = math.ceil(reduction / chunk) * stride
    result = []
    for rank, tile in enumerate(tiles):
        first = count * rank // len(tiles)
        end = count * (rank + 1) // len(tiles)
        begin_column = min(columns, first * width)
        result.append(
            {
                "tile": tile,
                "rank": rank,
                "column_start": begin_column,
                "columns": min(columns, end * width) - begin_column,
                "byte_start": first * panel_bytes,
                "bytes": (end - first) * panel_bytes,
            }
        )
    return result


def head_ranges(heads, tiles):
    tiles = participants(tiles)
    if heads <= 0:
        raise ValueError("Gemma query head count must be positive")
    return [
        (heads * rank // len(tiles), heads * (rank + 1) // len(tiles))
        for rank in range(len(tiles))
    ]


def kv_sources(config, tiles):
    tiles = participants(tiles)
    count = config.num_hidden_layers - config.num_kv_shared_layers
    if count <= 0 or config.num_key_value_heads != 1:
        raise ValueError(
            "Gemma task partition requires a single KV head and producer layers"
        )
    kinds = config.layer_types
    if len(kinds) != config.num_hidden_layers:
        raise ValueError("Gemma layer types do not cover all layers")
    last = {
        kind: max(i for i, value in enumerate(kinds[:count]) if value == kind)
        for kind in set(kinds)
    }
    return [
        {
            "source": layer if layer < count else last[kind],
            "owner": tiles[(layer if layer < count else last[kind]) % len(tiles)],
            "producer": layer < count,
        }
        for layer, kind in enumerate(kinds)
    ]
