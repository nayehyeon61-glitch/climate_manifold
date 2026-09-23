"""ClimODE-style casewise scores, with an explicit train-only climatology.

Reference: Aalto-QuML/ClimODE e729d23, utils.py evaluation_*_mm and
evaluation_global.py. This is a custom-data protocol, not paper reproduction.
"""
import hashlib
import torch


class ClimODEMetrics:
    def __init__(self, schema, climatology, lead_hours):
        self.schema = schema
        self.climatology = climatology
        self.leads = list(map(float, lead_hours))
        coords = schema['variables'][0]['coords']
        weight = torch.cos(torch.deg2rad(torch.tensor(coords['lat'], dtype=torch.float64)))
        self.weight = weight[:, None].expand(-1, len(coords['lon']))
        self.weight = self.weight / self.weight.sum()
        self.count = 0
        self.sums, self.squares, self.counts = {}, {}, {}
        self.protocol = {
            'id': 'climode_casewise_train_climatology_physical.v1',
            'source_commit': 'e729d23e8799ce0e075699e76d60227d848d8d0c',
            'weighting': 'cos(latitude), normalized over latitude/longitude cells',
            'rmse': 'spatial weighted RMSE per origin and lead, then arithmetic mean over origins',
            'acc': 'subtract fixed training climatology, then unweighted spatial mean per field; latitude-weighted correlation per origin',
            'climatology': 'checkpoint training temporal mean per grid cell; never evaluation labels',
            'climatology_sha256': hashlib.sha256(climatology.numpy().tobytes()).hexdigest(),
            'crps': 'analytic Gaussian marginal CRPS in physical units, latitude weighted; null for deterministic forecasts',
            'std': 'population standard deviation across cases, not confidence interval or seed variation',
            'undefined_acc': 'null with valid counts; do not compare ACC on different valid subsets',
            'aggregate': 'pool casewise scores over origins and leads within each variable; no mixing physical units',
            'paper_reproduction': False,
        }

    def update(self, prediction, truth, gaussian_scores=None):
        # Inputs are de-normalized CPU float64 [origin,lead,variable,lat,lon].
        if not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
            raise FloatingPointError('Nonfinite fields in ClimODE metrics')
        error = prediction - truth
        p, y = prediction-self.climatology, truth-self.climatology
        p = p-p.mean((-2, -1), keepdim=True)
        y = y-y.mean((-2, -1), keepdim=True)
        dot = (p*y*self.weight).sum((-2, -1))
        denom = ((p.square()*self.weight).sum((-2, -1)) *
                 (y.square()*self.weight).sum((-2, -1))).sqrt()
        valid = denom > 1e-15
        acc = torch.where(valid, (dot/denom.clamp_min(1e-15)).clamp(-1, 1), torch.nan)
        terms = {'rmse': (error.square()*self.weight).sum((-2, -1)).sqrt(), 'acc': acc}
        if gaussian_scores is not None:
            terms['crps'] = (gaussian_scores*self.weight).sum((-2, -1))
        for key, value in terms.items():
            mask = torch.isfinite(value)
            safe = torch.where(mask, value, 0.)
            for dest, total in ((self.sums, safe.sum(0)), (self.squares, safe.square().sum(0)),
                                (self.counts, mask.sum(0))):
                dest[key] = dest.get(key, torch.zeros_like(total)) + total
        self.count += len(prediction)

    def result(self):
        rows = {}
        for i, variable in enumerate(self.schema['variables']):
            def summarize(j=None):
                result = {'case_count': self.count*(len(self.leads) if j is None else 1)}
                for key in ('rmse', 'acc', 'crps'):
                    def get(source):
                        return source[key][:, i].sum() if j is None else source[key][j, i]
                    count = int(get(self.counts)) if key in self.counts else 0
                    result[key+'_valid_cases'] = count
                    mean = get(self.sums)/count if count else None
                    result[key] = float(mean) if count else None
                    result[key+'_std'] = float((get(self.squares)/count-mean.square()).clamp_min(0).sqrt()) if count else None
                return result
            rows[variable['name']] = {
                'units': variable.get('attrs', {}).get('units', {'msl':'Pa', 't2m':'K', 'u10':'m/s', 'v10':'m/s'}.get(variable['name'])),
                'aggregate': summarize(),
                'by_lead': [dict(lead_hours=h, **summarize(j)) for j, h in enumerate(self.leads)],
            }
        return {'protocol': self.protocol, 'case_count': self.count, 'per_variable': rows}
