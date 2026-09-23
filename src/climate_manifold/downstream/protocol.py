"""Explicit separation of latent forecasting and decoded-grid auxiliary studies."""


def experiment_contract(config):
    cfg = vars(config) if not isinstance(config, dict) else config
    bridge = cfg['bridge']
    representation = 'raw' if bridge == 'raw' else cfg.get('representation', 'climate_manifold')
    training_mode = cfg.get('training_mode', 'frozen')
    primary = (cfg['model'] in ('mlp', 'neural_ode', 'persistence')
               and bridge in ('raw', 'latent') and cfg['anchor'] == 'none')
    return {
        'suite': 'primary' if primary else 'auxiliary',
        'representation': representation,
        'prediction_space': 'latent' if bridge == 'latent' else 'field',
        'path': ('encoder -> predictor -> decoder' if bridge == 'latent' else
                 'encoder -> decoder -> grid predictor' if bridge == 'decoded' else
                 'observations -> field predictor'),
        'training_mode': training_mode,
        'representation_frozen': bridge != 'raw' and training_mode == 'frozen',
        'origin_residual_bypass': cfg['anchor'] != 'none',
    }


def validate_experiment(config, requested):
    if requested not in ('primary', 'auxiliary'):
        raise ValueError('Experiment must be primary or auxiliary')
    contract = experiment_contract(config)
    if requested == 'primary' and contract['suite'] != 'primary':
        raise ValueError('Primary experiments require raw or encoder -> latent model -> decoder, '
                         'anchor=none and MLP/Neural ODE (raw persistence is allowed). '
                         'Use --experiment auxiliary for ClimODE, decoded grids or residual anchoring.')
    return {**contract, 'suite': requested}
