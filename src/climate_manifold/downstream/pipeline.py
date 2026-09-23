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

    def __post_init__(self):
        if self.model not in ('mlp','neural_ode','climode','persistence'):
            raise ValueError('Unknown downstream model')
        if self.bridge not in ManifoldBridge.MODES or self.anchor not in ('none','origin'):
            raise ValueError('Invalid bridge/anchor')
        if self.representation not in ('climate_manifold', 'plain_ae'):
            raise ValueError('Unknown representation')
        if self.representation == 'plain_ae' and (self.bridge != 'latent' or self.model not in ('mlp', 'neural_ode')):
            raise ValueError('Plain AE control requires a latent MLP or Neural ODE')
        if self.model in ('climode','persistence') and self.bridge == 'latent':
            raise ValueError('ClimODE/persistence require raw or decoded grids, not a reshaped global latent')
        if min(self.hidden_dim,self.ode_substeps,self.velocity_iterations) < 1:
            raise ValueError('Model widths/integration counts must be positive')
        if not math.isfinite(self.climode_step_hours) or not 0 < self.climode_step_hours <= 6:
            raise ValueError('ClimODE step_hours must be in (0,6]')


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
        self.bridge = ManifoldBridge(selected, config.bridge, config.anchor)
        dimension = self.bridge.dimension
        info_dim = math.prod(manifold.info_metadata['shape']) if manifold.info_metadata else 0
        if config.model in ('mlp','neural_ode'):
            self.predictor = HistoryPredictor(dimension,manifold.config.history_steps,config.hidden_dim,
                info_dim if config.condition_information and config.bridge=='raw' else 0, config.model, config.ode_substeps)
        elif config.model == 'climode':
            from .climode import ClimODEPredictor, validate_constants
            if constants is None:raise ValueError('ClimODE requires real aligned orography and land-sea mask (--constants)')
            if schema is None:raise ValueError('ClimODE construction requires the archive schema')
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
            # Latent models can only receive dynamic information through the
            # frozen encoder, never a parallel raw-information input.
            direct_information = information if self.config.bridge == 'raw' else None
            predicted,std = self.predictor(features,lead_hours,origin_ns,direct_information)
        mean = self.bridge.to_fields(predicted,history[:,-1],features[:,-1])
        if not torch.isfinite(mean).all() or (std is not None and (not torch.isfinite(std).all() or (std <= 0).any())):
            raise FloatingPointError('Nonfinite or invalid downstream prediction')
        reconstruction = self.bridge.decode(features[:,-1])
        return {'mean':mean,'std':std,'reconstructed_origin':reconstruction,
                'predicted_latent':predicted if self.config.bridge == 'latent' else None,
                'origin_latent':features[:,-1] if self.config.bridge == 'latent' else None}
