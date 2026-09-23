"""DCT and MLP building blocks used by the manifold."""
import math
import torch
from torch import nn

class OrthoDCT(nn.Module):
    """Orthonormal DCT-II along one axis; inverse is exactly the transpose."""
    def __init__(self, size: int):
        super().__init__()
        n = torch.arange(size, dtype=torch.float64)
        matrix = torch.cos(math.pi / size * n[:, None] * (n[None, :] + 0.5))
        matrix *= math.sqrt(2.0 / size)
        matrix[0] /= math.sqrt(2.0)
        self.register_buffer("matrix", matrix)

    def forward(self, values, *, inverse=False, dim=-1):
        moved = values.movedim(dim, -1)
        matrix = self.matrix.to(dtype=values.dtype)
        result = moved @ (matrix if inverse else matrix.T)
        return result.movedim(-1, dim)


class FieldDCT(nn.Module):
    """Spatial DCT independently for every variable, never across variable names."""
    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.latitude = OrthoDCT(grid[1])
        self.longitude = OrthoDCT(grid[2])

    def forward(self, flat, *, inverse=False):
        shaped = flat.reshape(*flat.shape[:-1], *self.grid)
        shaped = self.latitude(shaped, inverse=inverse, dim=-2)
        shaped = self.longitude(shaped, inverse=inverse, dim=-1)
        return shaped.reshape_as(flat)


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                         nn.SiLU(), nn.Linear(hidden_dim, output_dim))
