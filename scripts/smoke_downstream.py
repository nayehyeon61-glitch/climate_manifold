"""Six direct/manifold model variants on a synthetic frozen A checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from smoke_climate_manifold import main as smoke_a
from climate_manifold.downstream.train import main as train_main
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.compare import compare
from climate_manifold.train import load_checkpoint


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--a-smoke-dir',help='Reuse a completed smoke_climate_manifold output directory')
    p.add_argument('--without-climode',action='store_true')
    args=p.parse_args(argv)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    if args.a_smoke_dir:source=Path(args.a_smoke_dir)
    else:
        source=output/'a';smoke_a(['--output',str(source)])
    checkpoint=source/'manifold.pt';archive=source/'synthetic-states.npz';info=source/'pinn-information.npz'
    _,meta=load_checkpoint(checkpoint)
    coords=meta['schema']['variables'][0]['coords'];shape=meta['schema']['variables'][0]['shape']
    constants=output/'synthetic-constants.npz'
    # Deliberately synthetic fixtures. Real runs must supply actual static fields.
    np.savez_compressed(constants,**coords,orography=np.full(shape,500.),lsm=np.full(shape,.5),orography_units='m')
    variants=[('mlp','raw'),('mlp','latent'),('neural_ode','raw'),('neural_ode','latent')]
    if not args.without_climode:variants += [('climode','raw'),('climode','decoded')]
    reports=[]
    for family,bridge in variants:
        prefix=output/(family+'-'+bridge)
        train_main(['--a-checkpoint',str(checkpoint),'--archive',str(archive),'--information',str(info),
            '--output',str(prefix)+'.pt','--model',family,'--bridge',bridge,'--constants',str(constants),
            '--epochs','1','--batch-size','2','--hidden-dim','24','--max-windows','2',
            '--horizon-steps','20','--window-stride','1','--no-climode-attention',
            '--climode-step-hours','6','--velocity-iterations','2'])
        report=str(prefix)+'.json'
        evaluate(str(prefix)+'.pt',str(archive),report,information=str(info),max_cases=2,
                 forecast_output=str(prefix)+'.npz')
        reports.append(report)
    result=compare(reports,output/'comparison.json')
    assert result['ranking_allowed'],result
    print(json.dumps({'software_smoke_only':True,'variants':len(result['rows']),'comparison':str(output/'comparison.json')}))
    return 0

if __name__=='__main__':raise SystemExit(main())
