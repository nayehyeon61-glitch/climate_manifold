"""Explicit separation of latent forecasting and decoded-grid auxiliary studies."""


def _sequence_supported(cfg):
    return (cfg.get('training_mode') == 'joint' and cfg.get('latent_layout') == 'spatial'
            and cfg.get('anchor') == 'none' and cfg.get('representation','climate_manifold') == 'climate_manifold'
            and (cfg['bridge'] == 'latent'
                 or cfg['bridge'] == 'raw' and cfg.get('raw_backend') == 'matched'))


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
    sequence = cfg['model'] in ('convlstm', 'simvp') and _sequence_supported(cfg)
    primary = ((cfg['model'] in ('mlp', 'neural_ode', 'persistence') or latent_climode or matched_raw or sequence)
               and bridge in ('raw', 'latent') and cfg['anchor'] == 'none')
    return {
        'suite': 'primary' if primary else 'auxiliary',
        'representation': representation,
        'prediction_space': 'latent' if bridge == 'latent' else 'field',
        'path': ('encoder -> predictor -> decoder' if bridge == 'latent' else
                 'encoder -> decoder -> grid predictor' if bridge == 'decoded' else
                 'observations -> field predictor'),
        'training_mode': training_mode,
        'latent_layout': latent_layout if bridge == 'latent' else None,
        'raw_backend': cfg.get('raw_backend', 'legacy') if bridge == 'raw' else None,
        'predictor_variant': ('raw_transport_climode' if matched_raw and cfg['model'] == 'climode' else
                              'raw_spatial_' + cfg['model'] if matched_raw else
                              'latent_transport_climode' if latent_climode else
                              'spatial_' + cfg['model'] if bridge == 'latent' and latent_layout == 'spatial' else
                              cfg['model']),
        'representation_frozen': bridge != 'raw' and training_mode == 'frozen',
        'origin_residual_bypass': cfg['anchor'] != 'none',
    }


def validate_experiment(config, requested):
    if requested not in ('primary', 'auxiliary'):
        raise ValueError('Experiment must be primary or auxiliary')
    cfg = vars(config) if not isinstance(config, dict) else config
    if cfg['model'] in ('convlstm', 'simvp') and not _sequence_supported(cfg):
        raise ValueError('ConvLSTM/SimVP require joint spatial latent or matched raw forecasting, anchor=none')
    contract = experiment_contract(config)
    if requested == 'primary' and contract['suite'] != 'primary':
        raise ValueError('Primary experiments require raw or encoder -> latent model -> decoder, '
                         'anchor=none and MLP/Neural ODE or joint spatial latent/matched-raw ClimODE/ConvLSTM/SimVP '
                         '(raw persistence is allowed). Use --experiment auxiliary for legacy raw '
                         'or decoded ClimODE, decoded grids or residual anchoring.')
    return {**contract, 'suite': requested}
