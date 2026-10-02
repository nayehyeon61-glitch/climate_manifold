"""Compose causal preprocessing, a downstream model, and physical-field outputs."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .bridge import ManifoldBridge
from .baselines import HistoryPredictor


SEQUENCE_IMPLEMENTATIONS = {
    'convlstm': 'convlstm_time_conditioned_adaptation_v1',
    'simvp': 'simvp_gsta_lead_conditioned_adaptation_v1',
}
WEATHER_IMPLEMENTATIONS = {
    'fourcastnet': 'fourcastnet_afno_context_adaptation_v1',
    'climax': 'climax_variable_token_adaptation_v1',
}
SPATIAL_IMPLEMENTATIONS = {**SEQUENCE_IMPLEMENTATIONS, **WEATHER_IMPLEMENTATIONS}
TRANSFORMER_IMPLEMENTATION = 'raw_latent_guide_transformer_v1'


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
    weather_depth: int = 4
    weather_patch_size: int = 2
    transformer_heads: int = 4
    guide_mode: str = 'learned'
    # Guided Transformers historically saw information only through E. When
    # enabled, origin information also enters as raw-matched tokens, so guided
    # and raw arms share the same direct information access.
    guide_direct_information: bool = False

    def __post_init__(self):
        if self.model not in ('mlp','neural_ode','climode','persistence','transformer',*SPATIAL_IMPLEMENTATIONS):
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
        if self.guide_mode not in ('learned', 'zero'):
            raise ValueError('Guide mode must be learned or zero')
        if self.guide_mode != 'learned' and self.bridge != 'guided':
            raise ValueError('Guide ablations require bridge=guided')
        if not isinstance(self.guide_direct_information, bool):
            raise ValueError('guide_direct_information must be a boolean')
        if self.guide_direct_information and (self.bridge != 'guided' or not self.condition_information):
            raise ValueError('Direct guide information requires bridge=guided with information conditioning')
        if (isinstance(self.transformer_heads, bool) or not isinstance(self.transformer_heads, int)
                or self.transformer_heads < 1):
            raise ValueError('Transformer heads must be a positive integer')
        if self.model == 'transformer':
            if (self.training_mode != 'joint' or self.latent_layout != 'spatial'
                    or self.bridge not in ('raw', 'latent', 'guided')
                    or self.bridge == 'raw' and self.raw_backend != 'matched'):
                raise ValueError('Transformer requires joint spatial latent/guided or matched raw forecasting')
            if self.hidden_dim % self.transformer_heads:
                raise ValueError('Transformer hidden_dim must be divisible by transformer_heads')
        if self.bridge == 'guided' and (self.model != 'transformer' or self.representation != 'climate_manifold'):
            raise ValueError('Guided bridge requires the Transformer and climate_manifold representation')
        if self.bridge == 'guided' and not self.condition_information:
            raise ValueError('Guided bridge encodes origin information; disabling information conditioning is unsupported')
        if self.model in SPATIAL_IMPLEMENTATIONS:
            if (self.training_mode != 'joint' or self.latent_layout != 'spatial'
                    or self.bridge not in ('raw', 'latent')
                    or self.bridge == 'raw' and self.raw_backend != 'matched'):
                raise ValueError('Spatial sequence/weather models require joint spatial latent or matched raw forecasting')
        if self.raw_backend == 'matched' and self.bridge == 'raw':
            if (self.training_mode != 'joint' or self.latent_layout != 'spatial'
                    or self.model not in ('mlp', 'neural_ode', 'climode', 'transformer', *SPATIAL_IMPLEMENTATIONS)):
                raise ValueError('Matched raw controls require a supported joint spatial predictor')
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
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.weather_depth, self.weather_patch_size)):
            raise ValueError('Weather depth and patch size must be positive integers')
        if not math.isfinite(self.climode_step_hours) or not 0 < self.climode_step_hours <= 6:
            raise ValueError('ClimODE step_hours must be in (0,6]')
        if any(not math.isfinite(v) or v<=0 for v in (self.latent_max_speed,self.latent_max_acceleration)):
            raise ValueError('Latent transport bounds must be finite and positive')


class ForecastPipeline(nn.Module):
    def __init__(self, manifold, config, constants=None, schema=None, representation=None,
                 *, separate_reconstruction_decoder=False, conditional_flow_config=None,
                 constraint_decoder_mode='separate_surface_and_information'):
        super().__init__()
        if not isinstance(separate_reconstruction_decoder, bool):
            raise ValueError('separate_reconstruction_decoder must be a boolean')
        if separate_reconstruction_decoder and (
                config.training_mode != 'joint' or config.bridge not in ('latent', 'guided')
                or config.anchor != 'none' or config.representation != 'climate_manifold'):
            raise ValueError('Separate reconstruction decoder requires a joint unanchored latent climate manifold')
        if conditional_flow_config is not None:
            if (config.training_mode != 'joint' or config.bridge != 'latent'
                    or config.anchor != 'none' or config.representation != 'climate_manifold'):
                raise ValueError('Conditional flow requires a joint unanchored latent climate manifold')
            if constraint_decoder_mode not in ('information_only', 'separate_surface_and_information'):
                raise ValueError('Conditional flow requires independent auxiliary decoders')
            if (constraint_decoder_mode == 'separate_surface_and_information'
                    and not separate_reconstruction_decoder):
                raise ValueError('Conditional flow requires a dedicated reconstruction_decoder')
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
        if config.bridge in ('latent', 'guided') and config.latent_layout != actual_layout:
            raise ValueError('Predictor latent layout does not match the actual manifold representation')
        self.bridge = ManifoldBridge(selected, config.bridge, config.anchor, config.training_mode)
        if config.bridge == 'guided':
            # Forecasts are physical fields produced by the Transformer head.
            # Preserve this dormant decoder for checkpoints, but never train it
            # on duplicate reconstruction or use it in the guided forecast.
            selected.core.manifold.decoder.requires_grad_(False)
        dimension = self.bridge.dimension
        info_dim = math.prod(manifold.info_metadata['shape']) if manifold.info_metadata else 0
        matched_raw = config.bridge == 'raw' and config.raw_backend == 'matched'
        information_channels = 0
        if matched_raw:
            if schema is None:
                raise ValueError('Matched raw spatial controls require the archive schema')
            from .latent_climode import _pooled_coordinates
            _, _, periodic_lon = _pooled_coordinates(manifold.config.grid, schema, 1)
        if matched_raw or config.guide_direct_information:
            if config.condition_information and manifold.info_metadata:
                shape = tuple(manifold.info_metadata['shape'])
                if len(shape) != 3 or shape[1:] != tuple(manifold.config.grid[1:]):
                    raise ValueError('Raw origin information must share the source spatial grid')
                information_channels = shape[0]
        if config.model == 'transformer':
            from .guided_transformer import GuidedTransformerPredictor
            self.predictor = GuidedTransformerPredictor(
                selected.config.latent_grid if config.bridge == 'latent' else selected.config.grid,
                guide_grid=selected.config.latent_grid if config.bridge == 'guided' else None,
                history_steps=selected.config.history_steps, hidden=config.hidden_dim,
                depth=config.weather_depth, patch_size=config.weather_patch_size,
                heads=config.transformer_heads, guide_mode=config.guide_mode,
                history_dt_hours=selected.config.history_stride*selected.config.step_hours,
                periodic_lon=periodic_lon if matched_raw else selected.core.physics.periodic_lon,
                information_channels=information_channels)
        elif config.model in ('mlp','neural_ode'):
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
        elif config.model in WEATHER_IMPLEMENTATIONS:
            if config.model == 'fourcastnet':
                from .fourcastnet import FourCastNetPredictor
                constructor = FourCastNetPredictor
                extra = {'forecast_step_hours': selected.config.step_hours}
            else:
                from .climax import ClimaXPredictor
                constructor = ClimaXPredictor
                extra = {'variable_names': ([v['name'] for v in schema['variables']]
                                            if matched_raw else None)}
            self.predictor = constructor(
                selected.config.grid if matched_raw else selected.config.latent_grid,
                selected.config.history_steps, hidden=config.hidden_dim,
                history_dt_hours=selected.config.history_stride*selected.config.step_hours,
                periodic_lon=periodic_lon if matched_raw else selected.core.physics.periodic_lon,
                information_channels=information_channels, depth=config.weather_depth,
                patch_size=config.weather_patch_size, **extra)
        elif config.model in SEQUENCE_IMPLEMENTATIONS:
            if config.model == 'convlstm':
                from .convlstm import ConvLSTMPredictor
                constructor = ConvLSTMPredictor
            else:
                from .simvp import SimVPPredictor
                constructor = SimVPPredictor
            self.predictor = constructor(
                selected.config.grid if matched_raw else selected.config.latent_grid,
                selected.config.history_steps, hidden=config.hidden_dim,
                history_dt_hours=selected.config.history_stride*selected.config.step_hours,
                periodic_lon=periodic_lon if matched_raw else selected.core.physics.periodic_lon,
                information_channels=information_channels)
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
        # Create the observed-field head after F so enabling it cannot change
        # the forecast model's random initialization. Historical checkpoints
        # omit it completely and retain their original state-dict contract.
        from .observed_decoder import ObservedFieldDecoder
        self.reconstruction_decoder = (
            ObservedFieldDecoder(selected) if separate_reconstruction_decoder else None)
        if self.reconstruction_decoder is not None and config.bridge == 'guided':
            self.reconstruction_decoder.requires_grad_(True)
        # This independent auxiliary vector field sees only the observed pair.
        # Construct it last so enabling CFM preserves the forecast initialization
        # and omit its parameters entirely in existing/off checkpoints.
        self.conditional_flow = None
        if conditional_flow_config is not None:
            from .conditional_flow import ObservedConditionalFlow
            self.conditional_flow = ObservedConditionalFlow(
                selected, conditional_flow_config, constraint_decoder_mode)
        self.train(self.training)

    def forward(self, history, information, origin_ns, lead_hours, *, reconstruct_origin=True):
        if (lead_hours.ndim != 1 or not len(lead_hours) or not torch.isfinite(lead_hours).all()
                or lead_hours[0] <= 0 or not (lead_hours[1:] > lead_hours[:-1]).all()):
            raise ValueError('Lead hours must be finite, positive and strictly increasing')
        features = self.bridge.encode_history(history,information)
        if self.config.bridge == 'guided':
            direct_information = information if self.config.guide_direct_information else None
            mean, std = self.predictor(history, lead_hours, origin_ns, direct_information,
                                       guide_history=features)
            if not torch.isfinite(mean).all():
                raise FloatingPointError('Nonfinite guided downstream prediction')
            return {'mean': mean, 'std': std, 'reconstructed_origin': None,
                    'history_latent': features, 'predicted_latent': None,
                    'origin_latent': features[:, -1]}
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
        # Information-only constraints do not train the surface decoder on the
        # observed origin. Keep this optional diagnostic for legacy objectives
        # and no-grad evaluation, without building an unused training graph.
        reconstruction = self.bridge.decode(features[:,-1]) if reconstruct_origin else None
        return {'mean':mean,'std':std,'reconstructed_origin':reconstruction,
                'history_latent':features if self.config.bridge == 'latent' else None,
                'predicted_latent':predicted if self.config.bridge == 'latent' else None,
                'origin_latent':features[:,-1] if self.config.bridge == 'latent' else None}
