"""Compose causal preprocessing, a downstream model, and physical-field outputs."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .bridge import ManifoldBridge
from .baselines import HistoryPredictor


@dataclass(frozen=True)
class PredictorConfig:
    model: str = 'neural_ode'
    bridge: str = 'latent'
    anchor: str = 'none'
    hidden_dim: int = 128
    ode_substeps: int = 2
    condition_information: bool = True
    climode_attention: bool = True
    climode_step_hours: float = 1.
    velocity_iterations: int = 20
    representation: str = 'climate_manifold'
    # Missing fields in historical checkpoints retain their frozen semantics.
    training_mode: str = 'frozen'
    latent_layout: str = 'global'
    latent_max_speed: float = 2.
    latent_max_acceleration: float = 1.
    # Historical raw checkpoints used global MLPs or the original ClimODE
    # backend. Keep that layout unless a new matched spatial control is chosen.
    raw_backend: str = 'legacy'

    def __post_init__(self):
        if self.model not in ('mlp','neural_ode','climode','persistence'):
            raise ValueError('Unknown downstream model')
        if self.bridge not in ManifoldBridge.MODES or self.anchor not in ('none','origin'):
            raise ValueError('Invalid bridge/anchor')
        if self.representation not in ('climate_manifold', 'plain_ae'):
            raise ValueError('Unknown representation')
        if self.training_mode not in ('joint', 'frozen'):
            raise ValueError('Training mode must be joint or frozen')
        if self.latent_layout not in ('global', 'spatial'):
            raise ValueError('Latent layout must be global or spatial')
        if self.raw_backend not in ('legacy', 'matched'):
            raise ValueError('Raw backend must be legacy or matched')
        if self.raw_backend == 'matched' and self.bridge == 'raw':
            if (self.training_mode != 'joint' or self.latent_layout != 'spatial'
                    or self.model not in ('mlp', 'neural_ode', 'climode')):
                raise ValueError('Matched raw controls require joint spatial MLP, Neural ODE or ClimODE')
        if self.training_mode == 'joint' and (self.model == 'persistence' or self.anchor != 'none'):
            raise ValueError('Joint training requires a trainable predictor and anchor=none')
        if self.representation == 'plain_ae' and (self.bridge != 'latent' or self.model not in ('mlp', 'neural_ode')):
            raise ValueError('Plain AE control requires a latent MLP or Neural ODE')
        if self.representation == 'plain_ae' and self.latent_layout == 'spatial':
            raise ValueError('Standalone plain AE checkpoints do not support spatial latent representations')
        if self.model == 'persistence' and self.bridge == 'latent':
            raise ValueError('Persistence requires raw or decoded grids, not a reshaped global latent')
        if self.model == 'climode' and self.bridge == 'latent':
            if self.latent_layout != 'spatial':
                raise ValueError('Latent ClimODE requires a spatial representation, not a reshaped global latent')
            if self.training_mode != 'joint':
                raise ValueError('Latent ClimODE requires joint encoder/predictor/decoder training')
        if min(self.hidden_dim,self.ode_substeps,self.velocity_iterations) < 1:
            raise ValueError('Model widths/integration counts must be positive')
        if not math.isfinite(self.climode_step_hours) or not 0 < self.climode_step_hours <= 6:
            raise ValueError('ClimODE step_hours must be in (0,6]')
        if any(not math.isfinite(v) or v<=0 for v in (self.latent_max_speed,self.latent_max_acceleration)):
            raise ValueError('Latent transport bounds must be finite and positive')


class ForecastPipeline(nn.Module):
    def __init__(self, manifold, config, constants=None, schema=None, representation=None):
        super().__init__()
        self.config, self.a_config = config, manifold.config
        if config.representation == 'plain_ae':
            if representation is None:
                raise ValueError('Plain AE requires an independently trained representation checkpoint')
            if representation.config != manifold.config or representation.info_metadata != manifold.info_metadata:
                raise ValueError('Plain AE and A representation/input contracts differ')
        elif representation is not None:
            raise ValueError('Unexpected alternative representation')
        selected = representation if config.representation == 'plain_ae' else manifold
        actual_layout = getattr(selected.config, 'representation_kind', 'global')
        if config.bridge == 'latent' and config.latent_layout != actual_layout:
            raise ValueError('Predictor latent layout does not match the actual manifold representation')
        self.bridge = ManifoldBridge(selected, config.bridge, config.anchor, config.training_mode)
        dimension = self.bridge.dimension
        info_dim = math.prod(manifold.info_metadata['shape']) if manifold.info_metadata else 0
        matched_raw = config.bridge == 'raw' and config.raw_backend == 'matched'
        information_channels = 0
        if matched_raw:
            if schema is None:
                raise ValueError('Matched raw spatial controls require the archive schema')
            from .latent_climode import _pooled_coordinates
            _, _, periodic_lon = _pooled_coordinates(manifold.config.grid, schema, 1)
            if config.condition_information and manifold.info_metadata:
                shape = tuple(manifold.info_metadata['shape'])
                if len(shape) != 3 or shape[1:] != tuple(manifold.config.grid[1:]):
                    raise ValueError('Raw origin information must share the source spatial grid')
                information_channels = shape[0]
        if config.model in ('mlp','neural_ode'):
            if matched_raw:
                from .spatial_baselines import SpatialHistoryPredictor
                self.predictor = SpatialHistoryPredictor(manifold.config.grid,
                    manifold.config.history_steps, config.hidden_dim, config.model, config.ode_substeps,
                    periodic_lon=periodic_lon, information_channels=information_channels)
            elif config.bridge == 'latent' and config.latent_layout == 'spatial':
                from .spatial_baselines import SpatialHistoryPredictor
                self.predictor = SpatialHistoryPredictor(selected.config.latent_grid,
                    selected.config.history_steps, config.hidden_dim, config.model, config.ode_substeps,
                    periodic_lon=selected.core.physics.periodic_lon)
            else:
                self.predictor = HistoryPredictor(dimension,manifold.config.history_steps,config.hidden_dim,
                    info_dim if config.condition_information and config.bridge=='raw' else 0, config.model, config.ode_substeps)
        elif config.model == 'climode':
            if schema is None:raise ValueError('ClimODE construction requires the archive schema')
            if config.bridge == 'latent' or matched_raw:
                from .latent_climode import LatentClimODEPredictor
                # Same transport core as E--ClimODE--D, now acting directly on
                # normalized physical fields. Cell/day speeds are scaled with
                # grid resolution; this is still an adapted transport model,
                # not the original ClimODE backend or its uncertainty head.
                factor = 1 if matched_raw else selected.config.spatial_downsample
                speed = config.latent_max_speed * (selected.config.spatial_downsample if matched_raw else 1)
                self.predictor = LatentClimODEPredictor(selected.config.grid if matched_raw else selected.config.latent_grid, schema,
                    hidden=config.hidden_dim, step_hours=config.climode_step_hours,
                    history_dt_hours=selected.config.history_stride*selected.config.step_hours,
                    spatial_factor=factor, information_channels=information_channels,
                    max_speed=speed,max_acceleration=config.latent_max_acceleration)
            else:
                from .climode import ClimODEPredictor, validate_constants
                if constants is None:raise ValueError('ClimODE requires real aligned orography and land-sea mask (--constants)')
                constants = validate_constants(constants, schema)
                self.predictor = ClimODEPredictor(manifold.config.grid, constants,
                    attention=config.climode_attention,step_hours=config.climode_step_hours,
                    velocity_iterations=config.velocity_iterations,
                    history_dt_hours=manifold.config.history_stride*manifold.config.step_hours)
        else:self.predictor = None
        self.train(self.training)

    def forward(self, history, information, origin_ns, lead_hours):
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        features = self.bridge.encode_history(history,information)
        if self.predictor is None:
            predicted = features[:,-1,None].expand(-1,len(lead_hours),-1);std=None
        else:
            # Latent models receive information through their encoder, never a
            # parallel raw-information input. Decoded ClimODE remains a physical
            # grid model: gradients reach its initial field through E/D, while
            # its separate observed-history velocity fit deliberately detaches.
            direct_information = information if self.config.bridge == 'raw' else None
            if self.config.raw_backend == 'matched' and not self.config.condition_information:
                direct_information = None
            predicted,std = self.predictor(features,lead_hours,origin_ns,direct_information)
        if self.config.bridge == 'latent' and std is not None:
            raise ValueError('Latent variance cannot be treated as physical-field Gaussian variance through a nonlinear decoder')
        mean = self.bridge.to_fields(predicted,history[:,-1],features[:,-1])
        if not torch.isfinite(mean).all() or (std is not None and (not torch.isfinite(std).all() or (std <= 0).any())):
            raise FloatingPointError('Nonfinite or invalid downstream prediction')
        reconstruction = self.bridge.decode(features[:,-1])
        return {'mean':mean,'std':std,'reconstructed_origin':reconstruction,
                'history_latent':features if self.config.bridge == 'latent' else None,
                'predicted_latent':predicted if self.config.bridge == 'latent' else None,
                'origin_latent':features[:,-1] if self.config.bridge == 'latent' else None}
