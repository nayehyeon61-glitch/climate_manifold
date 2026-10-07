"""Explicit separation of latent forecasting and decoded-grid auxiliary studies."""


def _sequence_supported(cfg):
    return (cfg.get('training_mode') == 'joint' and cfg.get('latent_layout') == 'spatial'
            and cfg.get('anchor') == 'none' and cfg.get('representation','climate_manifold') == 'climate_manifold'
            and (cfg['bridge'] == 'latent'
                 or cfg['bridge'] == 'raw' and cfg.get('raw_backend') == 'matched'
                 or cfg['bridge'] == 'guided' and (cfg['model'] == 'transformer' or _guide_fusion(cfg))))


def _guide_fusion(cfg):
    return (cfg['bridge'] == 'guided' and cfg.get('guide_architecture', 'joint') == 'fusion'
            and cfg.get('raw_backend') == 'matched'
            and cfg['model'] in ('mlp', 'neural_ode', 'climode', 'convlstm', 'simvp', 'fourcastnet', 'climax', 'transformer'))


def experiment_contract(config):
    cfg = vars(config) if not isinstance(config, dict) else config
    bridge = cfg['bridge']
    representation = 'raw' if bridge == 'raw' else cfg.get('representation', 'climate_manifold')
    training_mode = cfg.get('training_mode', 'frozen')
    latent_layout = cfg.get('latent_layout', 'global')
    latent_climode = (cfg['model'] == 'climode' and bridge == 'latent'
                     and latent_layout == 'spatial' and training_mode == 'joint')
    matched_raw = (bridge == 'raw' and cfg.get('raw_backend', 'legacy') == 'matched'
                   and latent_layout == 'spatial' and training_mode == 'joint')
    sequence = cfg['model'] in ('convlstm', 'simvp', 'fourcastnet', 'climax', 'transformer') and _sequence_supported(cfg)
    fusion = _guide_fusion(cfg) and _sequence_supported(cfg)
    guided = (bridge == 'guided' and cfg['model'] == 'transformer' and not fusion and _sequence_supported(cfg))
    primary = ((cfg['model'] in ('mlp', 'neural_ode', 'persistence') or latent_climode or matched_raw or sequence or fusion)
               and bridge in ('raw', 'latent', 'guided') and cfg['anchor'] == 'none')
    contract = {
        'suite': 'primary' if primary else 'auxiliary',
        'representation': representation,
        'prediction_space': 'latent' if bridge == 'latent' else 'field',
        'path': ('encoder -> predictor -> decoder' if bridge == 'latent' else
                 'encoder -> fusion transformer(raw, guide) -> raw-grid predictor -> fields' if fusion else
                 'raw observations + encoded statistical guide -> transformer -> fields' if guided else
                 'encoder -> decoder -> grid predictor' if bridge == 'decoded' else
                 'observations -> field predictor'),
        'training_mode': training_mode,
        'latent_layout': latent_layout if bridge in ('latent', 'guided') else None,
        'raw_backend': cfg.get('raw_backend', 'legacy') if bridge == 'raw' else None,
        'predictor_variant': ('guide_fusion_raw_spatial_' + cfg['model'] if fusion else
                              'guided_transformer' if guided else
                              'raw_transport_climode' if matched_raw and cfg['model'] == 'climode' else
                              'raw_spatial_' + cfg['model'] if matched_raw else
                              'latent_transport_climode' if latent_climode else
                              'spatial_' + cfg['model'] if bridge == 'latent' and latent_layout == 'spatial' else
                              cfg['model']),
        'representation_frozen': bridge != 'raw' and training_mode == 'frozen',
        'origin_residual_bypass': cfg['anchor'] != 'none',
    }
    if cfg.get('dynamics_encoder', 'off') != 'off':
        contract.update(
            dynamics_encoder=cfg['dynamics_encoder'],
            dynamics_mode=cfg.get('dynamics_mode', 'learned'),
            path=('statistical encoder + dynamics encoder -> latent residual fusion -> predictor -> decoder'
                  if bridge == 'latent' else
                  'statistical encoder + dynamics encoder -> fusion transformer(raw, statistical guide, dynamics guide) -> raw-grid predictor -> fields'),
            predictor_variant=contract['predictor_variant'] + '_dual_encoder',
        )
    if cfg.get('guide_interaction', 'off') != 'off':
        contract.update(
            guide_interaction=cfg['guide_interaction'],
            guide_interaction_depth=cfg.get('guide_interaction_depth', 1),
            path=('statistical encoder + dynamics encoder -> guide interaction transformer(spatial, dynamics) '
                  '-> fusion transformer(raw, refined guides) -> raw-grid predictor -> fields'),
            predictor_variant=contract['predictor_variant'] + '_two_stage',
        )
    return contract


def dynamics_identity(row):
    """Off/missing metadata preserves the historical experiment identity."""
    cfg = row.get('config', row)
    encoder = cfg.get('dynamics_encoder', 'off')
    return (encoder, cfg.get('dynamics_mode', 'learned')) if encoder != 'off' else ('off', None)


def dynamics_supervision_identity(row):
    """Separate auxiliary architectures/objectives without changing encoder identity.

    Missing metadata is the historical direct future-field head. This identity is
    for matching experiments; latent errors never enter physical forecast ranks.
    """
    cfg = row.get('config', row)
    if dynamics_identity(row)[0] == 'off':
        return ('off', None, None)
    auxiliary = row.get('dynamics_auxiliary_config') or {}
    architecture = cfg.get('dynamics_auxiliary_architecture', 'direct')
    return (architecture, auxiliary.get('supervision', 'field'),
            auxiliary.get('latent_weight', 0.))


def guide_interaction_contract(config):
    """Describe the independent guide-only first attention stage."""
    cfg = vars(config) if not isinstance(config, dict) else config
    if cfg.get('guide_interaction', 'off') == 'off':
        return None
    return {
        'version': 'climate_manifold.guide_interaction.v1',
        'kind': cfg['guide_interaction'], 'depth': cfg.get('guide_interaction_depth', 1),
        'input': 'spatial_and_dynamics_guide_tokens', 'raw_input': False,
        'output': 'refined_guide_tokens', 'integration': 'before_raw_guide_fusion',
        'parameters_shared_with_fusion': False,
    }


def guide_interaction_identity(row):
    """Separate trained single/two-stage models from same-checkpoint bypasses."""
    cfg = row.get('config', row)
    kind = cfg.get('guide_interaction', 'off')
    if kind == 'off':
        return ('off', None, 'off')
    evaluation = row.get('guide_interaction_evaluation') or {}
    return (kind, cfg.get('guide_interaction_depth', 1), evaluation.get('effective', kind))


def validate_experiment(config, requested):
    if requested not in ('primary', 'auxiliary'):
        raise ValueError('Experiment must be primary or auxiliary')
    cfg = vars(config) if not isinstance(config, dict) else config
    auxiliary_architecture = cfg.get('dynamics_auxiliary_architecture', 'direct')
    if auxiliary_architecture not in ('direct', 'latent_rollout'):
        raise ValueError('dynamics_auxiliary_architecture must be direct or latent_rollout')
    if auxiliary_architecture != 'direct' and cfg.get('dynamics_encoder', 'off') == 'off':
        raise ValueError('dynamics_auxiliary_architecture=latent_rollout requires dynamics encoder')
    interaction = cfg.get('guide_interaction', 'off')
    if interaction not in ('off', 'transformer'):
        raise ValueError('guide_interaction must be off or transformer')
    if interaction != 'off':
        depth = cfg.get('guide_interaction_depth', 1)
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError('guide_interaction_depth must be a positive integer')
        if not _guide_fusion(cfg) or cfg.get('dynamics_encoder', 'off') == 'off':
            raise ValueError('guide_interaction requires guided fusion and an enabled dynamics encoder')
    if cfg['model'] in ('convlstm', 'simvp', 'fourcastnet', 'climax', 'transformer') and not _sequence_supported(cfg):
        raise ValueError('Spatial sequence/weather models require joint spatial latent or matched raw forecasting, anchor=none')
    if cfg['bridge'] == 'guided' and (not (cfg['model'] == 'transformer' or _guide_fusion(cfg))
                                      or not _sequence_supported(cfg)):
        raise ValueError('Guided forecasts require a joint spatial climate_manifold transformer or guide fusion, anchor=none')
    if cfg.get('dynamics_encoder', 'off') != 'off':
        if (cfg.get('dynamics_encoder') not in ('conv_gru', 'temporal_conv')
                or cfg.get('dynamics_mode', 'learned') not in ('learned', 'zero')
                or cfg.get('training_mode') != 'joint' or cfg.get('latent_layout') != 'spatial'
                or cfg.get('representation', 'climate_manifold') != 'climate_manifold'
                or cfg.get('anchor') != 'none'
                or not (cfg['bridge'] == 'latent' or _guide_fusion(cfg))):
            raise ValueError('Dynamics encoder requires joint spatial climate_manifold latent or guided fusion, anchor=none')
    contract = experiment_contract(config)
    if requested == 'primary' and contract['suite'] != 'primary':
        raise ValueError('Primary experiments require raw or encoder -> latent model -> decoder, '
                         'anchor=none and MLP/Neural ODE or joint spatial latent/matched-raw ClimODE/ConvLSTM/SimVP/FourCastNet/ClimaX '
                         '(raw persistence is allowed). Use --experiment auxiliary for legacy raw '
                         'or decoded ClimODE, decoded grids or residual anchoring.')
    return {**contract, 'suite': requested}
