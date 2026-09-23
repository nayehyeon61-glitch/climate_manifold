"""Synthetic A/PINN multi-step dynamics -> reload -> pure/anchored forecast, on CPU."""
import argparse
import json
from pathlib import Path
import time
import torch
from synthetic_data import synthetic_archive, synthetic_pinn_information
from climate_manifold.train import main as train_main, load_checkpoint, write_json
from climate_manifold.forecast import evaluate
from climate_manifold.audit import audit
from climate_manifold.dynamics_evaluate import evaluate as evaluate_dynamics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--tiny',action='store_true',help='Use r=4, hidden=24 instead of the actual r=64, hidden=512')
    args=parser.parse_args(argv)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);started=time.perf_counter()
    archive,_=synthetic_archive(output,count=320)
    information=synthetic_pinn_information(archive,output)
    dims=(4,24,8) if args.tiny else (64,512,64)
    checkpoint=output/'manifold.pt'
    train_main(['--archive',str(archive),'--information',str(information),'--output',str(checkpoint),
        '--epochs','7','--curriculum-interval','1','--pinn','--pinn-warmup-epochs','1',
        '--pinn-ramp-epochs','2','--members','2','--tau-steps','1','--batch-size','2',
        '--history-stride','1','--max-windows','2','--window-stride','8',
        '--dynamics-max-steps','4',
        '--manifold-dim',str(dims[0]),'--hidden-dim',str(dims[1]),'--context-dim',str(dims[2])])
    model,payload=load_checkpoint(checkpoint)
    assert bool(model.core.manifold_ready)
    assert payload['dynamics_training']['enabled']
    assert payload['dynamics_training']['max_steps']==4
    assert not any(key.startswith(('core.experts.','core.gate.','core.history_encoder.')) for key in model.state_dict())
    epochs=json.loads(checkpoint.with_suffix('.metrics.json').read_text())
    train_steps=[int(row['train']['dynamics_rollout_steps']) for row in epochs
        if 'dynamics_rollout_steps' in row['train']]
    validation_steps=[int(row['validation']['dynamics_rollout_steps']) for row in epochs
        if 'dynamics_rollout_steps' in row['validation']]
    assert train_steps==[1,1,2,4,4,4],train_steps
    assert validation_steps==[4]*6,validation_steps
    reports={}
    for drift in (False,True):
        name='drift' if drift else 'auxiliary'
        reports[name]=evaluate(checkpoint,archive,output/(name+'.json'),information=information,
            members=2,tau_steps=1,max_cases=2,drift_only=drift,
            forecast_output=output/(name+'.npz'))['aggregate']
    pure=evaluate_dynamics(checkpoint,archive,output/'pure-drift.json',information=information,
        split='validation',steps=20,max_cases=2,forecast_output=output/'pure-drift.npz')
    assert pure['finite_forecast_fraction']==1.,pure['failed_origins']
    assert len(pure['origin_times'])==2
    assert pure['lead_hours']==list(range(6,121,6))
    assert pure['conditioning']['origin_residual_anchor'] is False
    assert pure['forecast_output_written']
    reports['pure_drift']=pure['scores']['aggregate']
    geometry=audit(checkpoint,archive,output/'geometry.json',information=information,max_pairs=4)
    write_json(output/'summary.json',{'check':'A/PINN multi-step dynamics end-to-end synthetic integration',
        'manifold_dim':dims[0],'hidden_dim':dims[1],'best_epoch':payload['best_epoch'],
        'parameter_count':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,
        'forecast':reports,'audited_pairs':geometry['unique_pairs'],
        'dynamics_training':payload['dynamics_training'],
        'training_rollout_steps':train_steps,'validation_rollout_steps':validation_steps,
        'pure_drift_validation_cases':len(pure['origin_times']),
        'pure_drift_validation_horizon_hours':pure['lead_hours'][-1],
        'limitation':'Software integration only; no ERA5 training, forecast skill, or calibration claim.'})
    print(output/'summary.json')
    return 0

if __name__=='__main__':raise SystemExit(main())
