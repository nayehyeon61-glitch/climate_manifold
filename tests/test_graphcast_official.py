"""Data-contract tests plus opt-in genuine upstream GraphCast gradient smoke."""
import json
import os
from types import SimpleNamespace
import numpy as np
import pytest
from climate_manifold.downstream.graphcast_official import (
    UPSTREAM_COMMIT, calendar_features, fit_upstream_normalization, load_checkpoint,
    make_batches, save_checkpoint, validate_grid, FORMAT)


def example():
    lat=np.array([-67.5,-22.5,22.5,67.5]);lon=np.arange(8)*45.
    schema={'forecast_step_hours':6,'state_dim':128,'variables':[
        {'name':name,'dims':['lat','lon'],'shape':[4,8],
         'coords':{'lat':lat.tolist(),'lon':lon.tolist()},'slice':[i*32,(i+1)*32]}
        for i,name in enumerate(['msl','t2m','u10','v10'])]}
    fields=np.arange(40*128,dtype=np.float32).reshape(40,128)
    data={'schema':schema,'states':fields,'times':np.datetime64('2000-01-01')+np.arange(40)*np.timedelta64(6,'h'),
          'train_end':20,'split':{'train':[0,1,2],'validation':[3,4]}}
    config=SimpleNamespace(history_span_steps=9,horizon_steps=3)
    return data,config


def test_observed_pair_uses_dense_six_hours_and_same_split_origin():
    data,config=example()
    indices,history,truth,hclock,fclock=next(make_batches(data,config,'validation',batch_size=2))
    np.testing.assert_array_equal(indices,[11,12])
    np.testing.assert_array_equal(history[0].reshape(2,-1),data['states'][[10,11]])
    np.testing.assert_array_equal(truth[0].reshape(3,-1),data['states'][[12,13,14]])
    assert hclock.shape==(2,2,4,8) and fclock.shape==(2,3,4,8)


def test_calendar_is_deterministic_known_future_and_longitude_dependent():
    times=np.array(['2000-01-01T00','2000-01-01T06'],dtype='datetime64[h]')
    features=calendar_features(times,np.array([0,90]))
    assert features.shape==(2,4,2)
    assert np.isfinite(features).all() and np.max(np.abs(features))<=1.
    np.testing.assert_allclose(features[0,2:4,1],features[1,2:4,0],atol=1e-6)


def test_statistics_never_use_heldout_fields():
    data,_=example();first=fit_upstream_normalization(data)
    data['states'][data['train_end']:]+=1e8
    assert fit_upstream_normalization(data)==first


def test_regional_or_nonperiodic_grid_is_rejected():
    data,_=example();validate_grid(data['schema'])
    for variable in data['schema']['variables']:
        variable['coords']['lon']=list(np.arange(8)*5.)
    with pytest.raises(ValueError,match='global longitude'):
        validate_grid(data['schema'])


def test_pickle_free_checkpoint_roundtrip_and_pin_guard(tmp_path):
    params={'gnn/linear':{'w':np.ones((2,3),np.float32),'b':np.zeros(3,np.float32)}}
    path=tmp_path/'checkpoint.npz'
    meta={'format':FORMAT,'upstream_commit':UPSTREAM_COMMIT}
    save_checkpoint(path,params,meta);restored,loaded=load_checkpoint(path)
    np.testing.assert_array_equal(restored['gnn/linear']['w'],params['gnn/linear']['w'])
    assert loaded['upstream_commit']==UPSTREAM_COMMIT
    save_checkpoint(path,params,{**meta,'upstream_commit':'different'})
    with pytest.raises(ValueError,match='compatible pinned'):
        load_checkpoint(path)


@pytest.mark.skipif(os.environ.get('GRAPHCAST_OFFICIAL_SMOKE')!='1',reason='Requires isolated official JAX environment')
def test_official_graphcast_rollout_has_finite_nonzero_gradients():
    from climate_manifold.downstream.graphcast_official import make_network,official_dependencies
    _,jax,jnp,*_=official_dependencies()
    data,_=example()
    rng=np.random.default_rng(7)
    history=rng.normal(size=(1,2,4,4,8)).astype(np.float32)
    hclock=calendar_features(data['times'][None,:2],np.arange(8)*45.)
    fclock=calendar_features(data['times'][None,2:4],np.arange(8)*45.)
    config=dict(resolution=45.,mesh_size=0,latent_size=8,gnn_msg_steps=1,hidden_layers=1,
                radius_query_fraction_edge_length=1.,mesh2grid_edge_normalization_factor=None)
    network,rollout=make_network(data['schema'],config,{'mean':[0]*4,'scale':[1]*4,'residual_scale':[1]*4})
    params=network.init(jax.random.PRNGKey(7),history,hclock,fclock[:,0])
    def loss(params):
        prediction=rollout(params,history,hclock,fclock)
        assert prediction.shape==(1,2,4,4,8)
        return jnp.mean(prediction**2)
    value,grads=jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(float(value))
    arrays=jax.tree_util.tree_leaves(grads)
    assert all(np.isfinite(np.asarray(x)).all() for x in arrays)
    assert sum(float(np.abs(np.asarray(x)).sum()) for x in arrays)>0
