"""Pooled, area-weighted physical-unit scores on identical forecast origins."""
import math
import numpy as np
import torch
from ..archive import field_grid
from ..temporal_supervision import area_weights
from .climode_metrics import ClimODEMetrics


def gaussian_crps(mean, std, truth):
    z = (truth-mean)/std
    cdf = .5*(1+torch.erf(z/math.sqrt(2)))
    density = torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    return std*(z*(2*cdf-1)+2*density-1/math.sqrt(math.pi))


class LatentDiagnostics:
    """Within-representation audits; future observations are diagnostic targets only."""

    def __init__(self, schema, lead_hours):
        c, _, _ = field_grid(schema)
        self.metric = torch.as_tensor(area_weights(schema), dtype=torch.float64).flatten().repeat(c)/c
        self.leads = np.asarray(lead_hours, dtype=float)
        self.dt = torch.tensor(np.diff(np.r_[0., self.leads]), dtype=torch.float64)[None, :, None]
        self.count = 0
        self.sums = {}

    def update(self, prediction, target_q, target_reconstruction, cycle_q, truth):
        values = [prediction['predicted_latent'], prediction['origin_latent'],
                  target_q, target_reconstruction, cycle_q, truth,
                  prediction['mean'], prediction['reconstructed_origin']]
        if not all(torch.isfinite(x).all() for x in values):
            raise FloatingPointError('Nonfinite latent reconstruction diagnostic')
        q, q0, target, reconstruction, cycle, truth, field, origin_field = [x.detach().cpu().double() for x in values]
        dq = torch.cat((q0[:, None], q), 1).diff(dim=1)/self.dt
        dy = torch.cat((q0[:, None], target), 1).diff(dim=1)/self.dt
        terms = {
            'latent_mse': (q-target).square().mean(-1),
            'latent_persistence_mse': (q0[:, None]-target).square().mean(-1),
            'latent_tendency_mse': (dq-dy).square().mean(-1),
            'predicted_tendency2': dq.square().mean(-1),
            'target_tendency2': dy.square().mean(-1),
            'latent_cycle_mse': (q-cycle).square().mean(-1),
            'future_reconstruction_mse': ((reconstruction-truth).square()*self.metric).sum(-1),
            'projected_persistence_mse': ((origin_field[:, None]-truth).square()*self.metric).sum(-1),
            'forecast_vs_reconstructed_target_mse': ((field-reconstruction).square()*self.metric).sum(-1),
        }
        for key, value in terms.items():
            total = value.sum(0)
            self.sums[key] = self.sums.get(key, torch.zeros_like(total))+total
        self.count += len(q)

    def result(self):
        if not self.count:
            return {'case_count': 0, 'aggregate': None, 'by_lead': []}
        s = {key:value/self.count for key,value in self.sums.items()}
        def summarize(index=None):
            get = lambda key:s[key].mean() if index is None else s[key][index]
            names = {'latent_rmse':'latent_mse', 'latent_persistence_rmse':'latent_persistence_mse',
                     'latent_tendency_rmse_per_hour':'latent_tendency_mse', 'latent_cycle_rmse':'latent_cycle_mse',
                     'future_reconstruction_normalized_rmse':'future_reconstruction_mse',
                     'projected_persistence_normalized_rmse':'projected_persistence_mse',
                     'forecast_vs_reconstructed_target_normalized_rmse':'forecast_vs_reconstructed_target_mse'}
            result = {name:float(get(key).sqrt()) for name,key in names.items()}
            amplitude = get('target_tendency2').sqrt()
            result['latent_tendency_amplitude_ratio'] = (float(get('predicted_tendency2').sqrt()/amplitude)
                                                        if amplitude > 1e-15 else None)
            return result
        return {'case_count':self.count, 'aggregate':summarize(),
                'by_lead':[dict(lead_hours=float(h), **summarize(i)) for i,h in enumerate(self.leads)],
                'limits':'Latent errors use representation-specific standardized coordinates and cannot rank different encoders. '
                         'Future reconstruction uses observed future fields with origin information only; it is not a forecast '
                         'or a rigorous forecast-error lower bound. Cycle consistency does not guarantee physical validity.'}


class ForecastMetrics:
    def __init__(self, schema, mean, scale, lead_hours):
        self.schema, self.grid = schema, field_grid(schema)
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(self.grid)
        self.scale = torch.as_tensor(scale, dtype=torch.float64).reshape(self.grid)
        self.area = torch.as_tensor(area_weights(schema), dtype=torch.float64)
        self.leads = np.asarray(lead_hours, dtype=float)
        self.dt = torch.tensor(np.diff(np.r_[0.,self.leads]), dtype=torch.float64)[None,:,None,None,None]
        self.count = 0
        self.probabilistic = None
        self.sums = {}
        self.climode = ClimODEMetrics(schema, self.mean, lead_hours)

    def update(self, predicted, truth, origin, std=None, reconstructed_origin=None):
        b, t, _ = predicted.shape
        if t != len(self.leads) or truth.shape != predicted.shape:
            raise ValueError('Metric prediction/target/lead shapes differ')
        probabilistic = std is not None
        if self.probabilistic is not None and self.probabilistic != probabilistic:
            raise ValueError('Cannot mix deterministic and Gaussian predictions in a report')
        self.probabilistic = probabilistic
        reshape = lambda x:x.detach().cpu().double().reshape(b,t,*self.grid)
        raw_pred, raw_true = reshape(predicted)*self.scale+self.mean, reshape(truth)*self.scale+self.mean
        raw_origin = origin.detach().cpu().double().reshape(b,*self.grid)*self.scale+self.mean
        error = raw_pred-raw_true
        persistence_error = raw_origin[:,None]-raw_true
        dp = torch.cat((raw_origin[:,None],raw_pred),1).diff(dim=1)/self.dt
        dy = torch.cat((raw_origin[:,None],raw_true),1).diff(dim=1)/self.dt
        anomaly_pred, anomaly_true = raw_pred-self.mean, raw_true-self.mean
        terms = dict(mse=error.square(),mae=error.abs(),bias=error,
                     normalized_mse=(error/self.scale).square(),
                     persistence_mse=persistence_error.square(),
                     persistence_normalized_mse=(persistence_error/self.scale).square(),
                     tendency_mse=(dp-dy).square(),pred_tendency2=dp.square(),true_tendency2=dy.square(),
                     acc_dot=anomaly_pred*anomaly_true,acc_p2=anomaly_pred.square(),acc_t2=anomaly_true.square())
        if std is not None:
            sigma = reshape(std)*self.scale
            if (sigma <= 0).any() or not torch.isfinite(sigma).all():raise ValueError('Invalid forecast standard deviation')
            terms.update(gaussian_crps=gaussian_crps(raw_pred,sigma,raw_true),variance=sigma.square(),
                gaussian_nll=.5*math.log(2*math.pi)+sigma.log()+.5*(error/sigma).square(),
                coverage80=(error.abs() <= 1.2815515655446004*sigma).double())
        self.climode.update(raw_pred, raw_true, terms.get('gaussian_crps'))
        for key, value in terms.items():
            total = (value*self.area).sum((-2,-1)).sum(0)
            self.sums[key] = self.sums.get(key,torch.zeros_like(total))+total
        field_mean_error = (error*self.area).sum((-2,-1)).square().sum(0)
        self.sums['field_mean_mse'] = self.sums.get('field_mean_mse',torch.zeros_like(field_mean_error))+field_mean_error
        if reconstructed_origin is not None:
            re = (reconstructed_origin.detach().cpu().double().reshape(b,*self.grid)*self.scale+self.mean-raw_origin).square()
            total = (re*self.area).sum((-2,-1)).sum(0)
            self.sums['reconstruction_mse'] = self.sums.get('reconstruction_mse',torch.zeros_like(total))+total
        names = [v['name'] for v in self.schema['variables']]
        if 'u10' in names and 'v10' in names:
            u,v = names.index('u10'),names.index('v10')
            speed_p = (raw_pred[:,:,u].square()+raw_pred[:,:,v].square()).sqrt()
            speed_t = (raw_true[:,:,u].square()+raw_true[:,:,v].square()).sqrt()
            total = ((speed_p-speed_t).square()*self.area).sum((-2,-1)).sum(0)
            self.sums['wind_speed_mse'] = self.sums.get('wind_speed_mse',torch.zeros_like(total))+total
        self.count += b

    @staticmethod
    def ratio(num,den):
        return float(num/den) if float(den)>1e-15 else None

    def result(self):
        if not self.count:return {'case_count':0,'per_variable':{},'aggregate':None,'climode':self.climode.result()}
        s = {k:v/self.count for k,v in self.sums.items()}
        rows = {}
        for i,variable in enumerate(self.schema['variables']):
            def summarize(j=None):
                get = lambda key: s[key][:,i].mean() if j is None else s[key][j,i]
                rmse = get('mse').sqrt()
                skill = self.ratio(get('mse').sqrt(),get('persistence_mse').sqrt())
                row = {'rmse':float(rmse),'mae':float(get('mae')),'bias':float(get('bias')),
                       'acc_train_mean':self.ratio(get('acc_dot'),(get('acc_p2')*get('acc_t2')).sqrt()),
                       'persistence_rmse':float(get('persistence_mse').sqrt()),
                       'rmse_skill_vs_persistence':None if skill is None else 1-skill,
                       'tendency_rmse_per_hour':float(get('tendency_mse').sqrt()),
                       'tendency_amplitude_ratio':self.ratio(get('pred_tendency2').sqrt(),get('true_tendency2').sqrt()),
                       'field_mean_rmse':float(get('field_mean_mse').sqrt())}
                if self.probabilistic:
                    row.update(gaussian_crps=float(get('gaussian_crps')),gaussian_nll=float(get('gaussian_nll')),
                               coverage80=float(get('coverage80')),spread=float(get('variance').sqrt()),
                               spread_skill_ratio=self.ratio(get('variance').sqrt(),rmse))
                return row
            rows[variable['name']] = {'units':variable.get('attrs',{}).get('units',{'msl':'Pa','t2m':'K','u10':'m/s','v10':'m/s'}.get(variable['name'])),
                'aggregate':summarize(),'by_lead':[dict(lead_hours=float(h),**summarize(j)) for j,h in enumerate(self.leads)]}
            if 'reconstruction_mse' in s:rows[variable['name']]['origin_reconstruction_rmse'] = float(s['reconstruction_mse'][i].sqrt())
        aggregate = {'normalized_rmse':float(s['normalized_mse'].mean().sqrt()),
                     'persistence_normalized_rmse':float(s['persistence_normalized_mse'].mean().sqrt())}
        if 'wind_speed_mse' in s:
            aggregate['wind_speed_rmse_mps'] = float(s['wind_speed_mse'].mean().sqrt())
        return {'case_count':self.count,'aggregate':aggregate,'per_variable':rows,'climode':self.climode.result(),
                'probabilistic_scores':'analytic pointwise Gaussian marginals' if self.probabilistic else 'not applicable; deterministic predictor',
                'acc_reference':'fixed per-grid training temporal mean, not seasonal/day-of-year climatology',
                'aggregation':'pool area-weighted squared errors over origins/leads before square root; overlapping origins are not independent'}
