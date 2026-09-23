"""Representation configuration and legacy global DCT autoencoder/drift."""
from __future__ import annotations
from dataclasses import dataclass
import math
from numbers import Integral, Real
import torch
from torch import nn
from .nn import FieldDCT, mlp
from .manifold_physics import SurfacePhysics

@dataclass(frozen=True)
class ManifoldConfig:
    state_dim: int
    grid: tuple[int, int, int]
    history_steps: int = 6
    history_stride: int = 4
    horizon_steps: int = 20
    step_hours: int = 6
    manifold_dim: int = 64
    hidden_dim: int = 512
    context_dim: int = 64
    residual_noise_std: float = 1.0  # raw intrinsic z / day, before sealing
    # Legacy checkpoints omit these fields and remain global representations.
    representation_kind: str = 'global'
    latent_channels: int = 32
    spatial_downsample: int = 2
    spatial_hidden_dim: int = 64

    def __post_init__(self):
        object.__setattr__(self, 'grid', tuple(self.grid))
        if (len(self.grid)!=3 or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in self.grid)
                or isinstance(self.state_dim, bool) or not isinstance(self.state_dim, Integral)
                or math.prod(self.grid)!=self.state_dim):
            raise ValueError('grid must be (variables, lat, lon) and multiply to state_dim')
        if self.representation_kind not in ('global', 'spatial'):
            raise ValueError('representation_kind must be global or spatial')
        names = ('history_steps', 'history_stride', 'horizon_steps', 'step_hours',
                 'manifold_dim', 'hidden_dim', 'context_dim', 'latent_channels',
                 'spatial_downsample', 'spatial_hidden_dim')
        if any(isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), Integral)
               or getattr(self, name) < 1 for name in names):
            raise ValueError('Positive integral dimensions are required')
        if self.representation_kind == 'spatial':
            if min(self.latent_grid[1:]) < 2:
                raise ValueError('Spatial downsampling must retain at least two latitude and longitude cells')
            object.__setattr__(self, 'manifold_dim', math.prod(self.latent_grid))
        elif self.manifold_dim >= self.state_dim:
            raise ValueError('Global manifold_dim < state_dim is required')
        if (isinstance(self.residual_noise_std, bool) or not isinstance(self.residual_noise_std, Real)
                or not math.isfinite(self.residual_noise_std) or self.residual_noise_std<=0):
            raise ValueError('residual_noise_std must be finite and positive')

    @property
    def latent_grid(self):
        if self.representation_kind != 'spatial':
            return None
        factor = self.spatial_downsample
        return (self.latent_channels, (self.grid[1]+factor-1)//factor,
                (self.grid[2]+factor-1)//factor)

    @property
    def history_span_steps(self):return (self.history_steps-1)*self.history_stride+1
    @property
    def horizon_hours(self):return self.horizon_steps*self.step_hours

class PhysicsManifoldAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_dct = FieldDCT(config.grid)
        self.encoder = mlp(config.state_dim, config.hidden_dim, config.manifold_dim)
        self.decoder = nn.Sequential(nn.Linear(config.manifold_dim, config.hidden_dim), nn.SiLU(),
                                     nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
                                     nn.Linear(config.hidden_dim, config.state_dim))
        self.latent_drift = mlp(config.manifold_dim, config.hidden_dim, config.manifold_dim)

    def encode(self, state):
        return self.encoder(self.spatial_dct(state))

    def decode(self, latent):
        return self.spatial_dct(self.decoder(latent), inverse=True)


class ManifoldCore(nn.Module):
    def __init__(self,config,schema,mean,scale):
        super().__init__()
        self.config=config
        self.manifold=PhysicsManifoldAE(config)
        self.physics=SurfacePhysics(schema,mean,scale)
        self.register_buffer('latent_mean',torch.zeros(config.manifold_dim))
        self.register_buffer('latent_scale',torch.ones(config.manifold_dim))
        self.register_buffer('manifold_ready',torch.tensor(False))

    def decode(self,q):
        return self.manifold.decode(q*self.latent_scale+self.latent_mean)

    def jacobian(self,q):
        with torch.inference_mode(False):
            normal_q=q.clone() if torch.is_inference(q) else q
            return torch.func.vmap(torch.func.jacfwd(self.decode))(normal_q)

def drift_per_day(core,q):
    return core.manifold.latent_drift(q*core.latent_scale+core.latent_mean)/core.latent_scale
