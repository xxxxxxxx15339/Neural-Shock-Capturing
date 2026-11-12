import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from experiments.accuracy_upgrade.train import canonical_inputs
from src.differentiable.euler_jax import get_euler_flux_jax
from src.differentiable.accuracy_upgrade import state_faces
from src.differentiable.teno5_jax import teno5_reconstruct_jax
from experiments.accuracy_upgrade.study import averages

def smooth(n): return jnp.asarray(averages(n,"coupled"))


def test_matched_inputs_are_actual_solver_inputs():
    u=smooth(16)
    for variant in ['B','C']:
        captured=[]
        def observe(s):
            captured.append(np.asarray(s))
            return teno5_reconstruct_jax(s)
        if variant=='B':get_euler_flux_jax(observe)(u)
        else:state_faces(u,observe)
        inputs=canonical_inputs(u,variant)
        assert len(inputs)==len(captured)==2
        for a,b in zip(inputs,captured):
            np.testing.assert_array_equal(a,b)
            # Same exact min-max normalization contract used by the backbone.
            def norm(x):
                x=np.asarray(x)
                span=np.ptp(x,axis=-1,keepdims=True)
                return (x-np.min(x,axis=-1,keepdims=True))/np.where(span>0,span,1.)
            np.testing.assert_array_equal(norm(a),norm(b))
