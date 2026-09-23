"""Explicit separation of latent forecasting and decoded-grid auxiliary studies."""


def experiment_contract(config):
    cfg = vars(config) if not isinstance(config, dict) else config
    bridge = cfg['bridge']
    representation = 'raw' if bridge == 'raw' else cfg.get('representation', 'climate_manifold')
    training_mode = cfg.get('training_mode', 'frozen')
    latent_layout = cfg.get('latent_layout', 'global')
    latent_climode = (cfg['model'] == 'climode' and bridge == 'latent'
                     and latent_layout == 'spatial' and training_mode == 'joint')
    primary = ((cfg['model'] in ('mlp', 'neural_ode', 'persistence') or latent_climode)
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
        'predictor_variant': ('latent_transport_climode' if latent_climode else
                              'spatial_' + cfg['model'] if bridge == 'latent' and latent_layout == 'spatial' else
                              cfg['model']),
        'representation_frozen': bridge != 'raw' and training_mode == 'frozen',
        'origin_residual_bypass': cfg['anchor'] != 'none',
    }


def validate_experiment(config, requested):
    if requested not in ('primary', 'auxiliary'):
        raise ValueError('Experiment must be primary or auxiliary')
    contract = experiment_contract(config)
    if requested == 'primary' and contract['suite'] != 'primary':
        raise ValueError('Primary experiments require raw or encoder -> latent model -> decoder, '
                         'anchor=none and MLP/Neural ODE or joint spatial latent ClimODE '
                         '(raw persistence is allowed). Use --experiment auxiliary for raw/decoded '
                         'ClimODE, decoded grids or residual anchoring.')
    return {**contract, 'suite': requested}
