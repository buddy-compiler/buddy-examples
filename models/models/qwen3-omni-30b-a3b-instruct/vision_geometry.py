import torch


def positions(grid, merge):
    result = []
    for temporal, height, width in grid.tolist():
        row, column = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        shape = (height // merge, merge, width // merge, merge)
        row = row.reshape(shape).transpose(1, 2).flatten()
        column = column.reshape(shape).transpose(1, 2).flatten()
        result.append(torch.stack((row, column), -1).repeat(temporal, 1))
    return torch.cat(result)


def interpolation(grid, side, merge):
    coordinates = positions(grid, merge).float()
    counts = grid.prod(-1)
    sizes = torch.repeat_interleave(grid[:, 1:], counts, dim=0)
    source = coordinates * (side - 1) / (sizes - 1).clamp(min=1)
    floor = source.floor()
    offsets = torch.arange(2)
    taps = (floor.long()[:, :, None] + offsets).clamp(0, side - 1)
    weights = (1 - (source[:, :, None] - floor[:, :, None] - offsets).abs()).clamp(
        min=0
    )
    indices = (taps[:, 0, :, None] * side + taps[:, 1, None, :]).reshape(-1, 4)
    coefficients = (weights[:, 0, :, None] * weights[:, 1, None, :]).reshape(-1, 4)
    return indices, coefficients
