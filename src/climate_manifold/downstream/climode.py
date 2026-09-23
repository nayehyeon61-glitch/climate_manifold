"""Adapted official ClimODE grid backend. See docs/downstream.md for deviations.

MIT-derived architecture/PDE: Aalto-QuML/ClimODE, e729d23e8799ce0e075699e76d60227d848d8d0c.
This is a custom-data experiment, not reproduction of the published five-variable benchmark.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def validate_constants(constants, schema):
    coords = schema['variables'][0]['coords']
    shape = tuple(schema['variables'][0]['shape'])
    for key in ('lat', 'lon'):
        if not np.array_equal(np.asarray(constants[key]), np.asarray(coords[key])):
            raise ValueError(f'ClimODE constants {key} must exactly match archive coordinates')
    for key in ('orography', 'lsm'):
        value = np.asarray(constants[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid ClimODE {key} grid')
    if np.any(np.asarray(constants['lsm']) < 0) or np.any(np.asarray(constants['lsm']) > 1):
        raise ValueError('Land-sea mask must be in [0,1]')
    if constants.get('orography_units') != 'm':
        raise ValueError('ClimODE constants require orography_units=m')
    return {k:np.asarray(v).tolist() if k != 'orography_units' else v for k,v in constants.items()}


def load_constants(path, schema):
    with np.load(path, allow_pickle=False) as f:
        value = {key:f[key] for key in ('lat','lon','orography','lsm')}
        value['orography_units'] = str(f['orography_units'])
    return validate_constants(value, schema)


def advection(state, velocity):
    c = state.shape[1]
    vx, vy = velocity[:, :c], velocity[:, c:]
    return (vx*torch.gradient(state, dim=3)[0] + vy*torch.gradient(state, dim=2)[0]
            + state*(torch.gradient(vx, dim=3)[0] + torch.gradient(vy, dim=2)[0]))


def fit_initial_velocity(history, history_dt_hours, iterations=20, ridge=1e-3):
    """Fit transport to the final observed backward difference; never future labels.

    Velocities use the upstream solver clock (one unit=100 hours). The ridge
    replaces the dense Gaussian-kernel precision used by the original scripts.
    These are variable-specific transport velocities, not observed 10m wind.
    """
    if history.shape[1] < 2 or history_dt_hours <= 0 or iterations < 1:
        raise ValueError('ClimODE velocity fit requires >=2 past fields and positive dt/iterations')
    field = history[:, -1].detach()
    target = ((history[:, -1]-history[:, -2])/(history_dt_hours*.01)).detach()
    with torch.enable_grad():
        velocity = nn.Parameter(field.new_zeros((len(field), 2*field.shape[1], *field.shape[-2:])))
        optimizer = torch.optim.Adam([velocity], lr=.1)
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = (advection(field, velocity)-target).square().mean() + ridge*velocity.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite causal velocity fit')
            loss.backward(); optimizer.step()
        return velocity.detach()


class ClimODEPredictor(nn.Module):
    def __init__(self, grid, constants, *, attention=True, step_hours=1., velocity_iterations=20,
                 history_dt_hours=24):
        super().__init__()
        from .._vendor.climode.model_function import Climate_encoder_free_uncertain, Climate_ResNet_2D
        from .._vendor.climode.model_utils import Self_attn_conv
        self.grid = tuple(grid)
        c, h, w = self.grid
        if min(h,w) < 2 or (attention and min(h,w) < 15):
            raise ValueError('ClimODE attention needs H,W>=15; disable attention explicitly for tiny tests')
        self.core = Climate_encoder_free_uncertain(c, 2, c, 'euler', attention, True, False)
        # Upstream input widths are hard-coded for C=5. Keep its architecture,
        # generalize only channel accounting for our canonical four surface fields.
        if c != 5:
            self.core.vel_f = Climate_ResNet_2D(5*c+39, [5,3,2], [128,64,2*c])
            if attention:self.core.vel_att = Self_attn_conv(5*c+39, 2*c)
            self.core.noise_net = Climate_ResNet_2D(c+38, [3,2,2], [128,64,2*c])
        lat, lon = np.meshgrid(constants['lat'], constants['lon'], indexing='ij')
        self.register_buffer('lat_map', torch.tensor(lat, dtype=torch.float32))
        self.register_buffer('lon_map', torch.tensor(lon, dtype=torch.float32))
        self.register_buffer('static', torch.tensor(np.stack((constants['orography'], constants['lsm'])), dtype=torch.float32)[None])
        self.step_hours, self.velocity_iterations = step_hours, velocity_iterations
        self.history_dt_hours = history_dt_hours

    def train(self, mode=True):
        super().train(mode)
        # A deterministic ODE RHS: dropout in the upstream residual blocks would
        # otherwise change the vector field at each numerical solver evaluation.
        for module in self.core.modules():
            if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):module.eval()
        return self

    def _one(self, history, lead_hours, origin_ns):
        from torchdiffeq import odeint
        _, _, c, h, w = history.shape
        # No persistent cache: caller may use different arrays with the same date.
        velocity = fit_initial_velocity(history, self.history_dt_hours, self.velocity_iterations)
        core = self.core
        core.new_lat_map = self.lat_map[None,None]*math.pi/180
        core.new_lon_map = self.lon_map[None,None]*math.pi/180
        core.lsm = self.static[:,1:2]
        core.oro = F.normalize(self.static[0,0], dim=1)[None,None]
        la, lo = core.new_lat_map, core.new_lon_map
        pos = torch.cat((la.cos(),lo.cos(),la.sin(),lo.sin(),la.sin()*lo.cos(),la.sin()*lo.sin()),1)
        final_pos = torch.cat((la,lo,pos,core.lsm,core.oro),1)
        # Bounded absolute clock prevents float32 loss of sub-hour precision.
        origin = np.datetime64(int(origin_ns), 'ns')
        year_start = origin.astype('datetime64[Y]').astype('datetime64[ns]')
        origin_hour = float((origin-year_start)/np.timedelta64(1,'h'))
        hours = torch.cat((lead_hours.new_zeros(1),lead_hours))
        initial = torch.cat((velocity,history[:,-1]),1)
        result = odeint(lambda t,x:core.pde(t+.01*origin_hour,x), initial, hours*.01,
                        method='euler', options={'step_size':.01*self.step_hours})
        states = result[:,:, -c:]
        # Upstream error network receives the same physical-hour clock as the RHS.
        mean, std = core.noise_net_contrib(hours+origin_hour, final_pos, states, core.noise_net,h,w)
        return mean[1:,0].flatten(1), std[1:,0].flatten(1).clamp_min(1e-4)

    def forward(self, history, lead_hours, origin_ns, information=None):
        fields = history.reshape(len(history),history.shape[1],*self.grid)
        result = [self._one(fields[i:i+1],lead_hours,int(origin_ns[i].detach().cpu())) for i in range(len(fields))]
        return torch.stack([x[0] for x in result]), torch.stack([x[1] for x in result])
