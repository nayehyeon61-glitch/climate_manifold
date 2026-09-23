"""Legacy frozen 3-way control; optional separate frozen ClimODE smoke."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from smoke_climate_manifold import main as smoke_a
from climate_manifold.downstream.plain_ae import main as train_plain_ae
from climate_manifold.downstream.train import main as train_main
from climate_manifold.downstream.evaluate import evaluate
from climate_manifold.downstream.compare import compare
from climate_manifold.train import load_checkpoint


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--a-smoke-dir',help='Reuse a completed smoke_climate_manifold output directory')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--with-climode',action='store_true',help='Also run two auxiliary grid comparisons')
    group.add_argument('--without-climode',action='store_true',help='Compatibility alias for the default primary-only smoke')
    args=p.parse_args(argv)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    if args.a_smoke_dir:source=Path(args.a_smoke_dir)
    else:
        source=output/'a';smoke_a(['--output',str(source)])
    checkpoint=source/'manifold.pt';archive=source/'synthetic-states.npz';info=source/'pinn-information.npz'
    ae_checkpoint=output/'plain-ae.pt'
    common=['--a-checkpoint',str(checkpoint),'--archive',str(archive),'--information',str(info)]
    train_plain_ae(common+['--output',str(ae_checkpoint),'--epochs','1','--batch-size','2',
                          '--max-windows','2','--window-stride','1'])
    reports=[]
    for family in ('mlp','neural_ode'):
        for variant in ('raw','climate_manifold','plain_ae'):
            bridge='raw' if variant=='raw' else 'latent'
            representation='climate_manifold' if variant=='raw' else variant
            extra=['--ae-checkpoint',str(ae_checkpoint)] if variant=='plain_ae' else []
            prefix=output/(family+'-'+variant)
            train_main(common+['--training-mode','frozen','--initialization','pretrained','--output',str(prefix)+'.pt','--experiment','primary',
                '--model',family,'--bridge',bridge,'--representation',representation]+extra+[
                '--epochs','1','--batch-size','2','--hidden-dim','24','--max-windows','2',
                '--horizon-steps','20','--window-stride','1'])
            report=str(prefix)+'.json'
            evaluate(str(prefix)+'.pt',str(archive),report,information=str(info),max_cases=2,
                     forecast_output=str(prefix)+'.npz')
            reports.append(report)
    result=compare(reports,output/'comparison.json')
    assert result['ranking_allowed'],result
    auxiliary_count=0
    if args.with_climode:
        _,meta=load_checkpoint(checkpoint)
        coords=meta['schema']['variables'][0]['coords'];shape=meta['schema']['variables'][0]['shape']
        constants=output/'synthetic-constants.npz'
        # Deliberately synthetic fixtures. Real runs must supply actual static fields.
        np.savez_compressed(constants,**coords,orography=np.full(shape,500.),lsm=np.full(shape,.5),orography_units='m')
        auxiliary=output/'auxiliary-climode';auxiliary.mkdir()
        auxiliary_reports=[]
        for bridge in ('raw','decoded'):
            prefix=auxiliary/('climode-'+bridge)
            train_main(common+['--training-mode','frozen','--initialization','pretrained','--output',str(prefix)+'.pt','--experiment','auxiliary',
                '--model','climode','--bridge',bridge,'--constants',str(constants),
                '--epochs','1','--batch-size','2','--hidden-dim','24','--max-windows','2',
                '--horizon-steps','20','--window-stride','1','--no-climode-attention',
                '--climode-step-hours','6','--velocity-iterations','2'])
            report=str(prefix)+'.json'
            evaluate(str(prefix)+'.pt',str(archive),report,information=str(info),max_cases=2,
                     forecast_output=str(prefix)+'.npz')
            auxiliary_reports.append(report)
        aux_result=compare(auxiliary_reports,auxiliary/'comparison.json')
        assert aux_result['ranking_allowed'],aux_result
        auxiliary_count=len(aux_result['rows'])
    print(json.dumps({'software_smoke_only':True,'primary_variants':len(result['rows']),
                      'auxiliary_variants':auxiliary_count,'comparison':str(output/'comparison.json')}))
    return 0

if __name__=='__main__':raise SystemExit(main())
