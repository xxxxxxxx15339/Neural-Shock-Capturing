"""Training uses real conservative SSPRK3 stages and rejects failed trajectories."""
import copy
import json
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.accuracy_upgrade.full_training import (
    conservative_rollout,loss_function,initial_params,read_config,build_update,save_model)
from src.differentiable.accuracy_upgrade import admissible
from src.differentiable.model_io import load_checkpoint
from src.differentiable.teno5_jax_toy import toy_keep_weight


def sod(n=16):
    left=jnp.array([1.,0.,2.5]);right=jnp.array([.125,0.,.25])
    return jnp.where((jnp.arange(n)<n//2)[:,None],left,right)


def test_loss_gradient_through_full_safeguard_and_balance():
    params=initial_params(4101);u=sod();dt=jnp.array(.001)
    final,d=jax.jit(lambda p:conservative_rollout(p,u,dt,steps=15))(params)
    assert d['valid'] and jnp.all(admissible(final))
    np.testing.assert_allclose(jnp.sum(final-u,axis=0)/len(u)+d['boundary_exchange'],0,atol=2e-15)
    def loss(p):return loss_function(p,u[None],u[None],dt[None],steps=15)[0]
    value,grad=jax.jit(jax.value_and_grad(loss))(params)
    assert jnp.isfinite(value)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    assert sum(float(jnp.sum(abs(x))) for x in jax.tree.leaves(grad))>0


def test_zero_neural_strength_has_classical_identity():
    params=initial_params(4102);u=sod()
    a,da=conservative_rollout(params,u,.001,steps=2,strength=0.)
    b,db=conservative_rollout(params,u,.001,steps=2,neural=False)
    assert da['valid'] and db['valid']
    np.testing.assert_array_equal(a,b)


def test_failed_batched_rollout_rejects_optimizer_update():
    config=copy.deepcopy(read_config());config['rollout_steps']=2
    optimizer,update=build_update(config);params=initial_params(4101);state=optimizer.init(params)
    good=sod();invalid=good.at[0,0].set(-1.)
    states=jnp.stack((invalid,good));target=jnp.stack((good,good));dt=jnp.array([.001,.001])
    result,next_state,loss,norm,accepted,diag=update(params,state,states,target,dt)
    assert not accepted and not diag['valid'][0]
    assert jnp.isfinite(loss) and jnp.isfinite(norm)
    for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(result)):np.testing.assert_array_equal(a,b)
    for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(next_state)):np.testing.assert_array_equal(a,b)


def test_new_checkpoint_compatible_and_deterministic(tmp_path):
    params=initial_params(4101);path=tmp_path/'conservative.pkl'
    save_model(path,params,dict(architecture=[5,64,32,16,8,3],seed=4101))
    checkpoint=load_checkpoint(path)
    s=jnp.array([[0.,0.,1.,1.,1.],[.1,.2,.1,0.,-.1]])
    np.testing.assert_array_equal(toy_keep_weight(params,s),toy_keep_weight(checkpoint.backbone,s))
    assert json.loads(path.with_suffix('.json').read_text())['sha256']==checkpoint.sha256
    with pytest.raises(FileExistsError):save_model(path,params,{})
