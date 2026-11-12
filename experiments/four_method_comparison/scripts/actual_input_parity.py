"""Check the original HDF5 adapter on inputs extracted from real saved rollouts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
OUT=ROOT/'experiments/four_method_comparison';RAW=OUT/'results/raw'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','keras','verify']);a=p.parse_args()
    if a.action=='keras':
        from keras.models import load_model
        model=load_model(str(ROOT/'models/nn_teno5.h5'),compile=False)
        with np.load(RAW/'actual_classifier_inputs.npz') as z:raw=z['raw']
        lo=raw.min(axis=-1,keepdims=True);span=np.ptp(raw,axis=-1,keepdims=True)
        normalized=(raw-lo)/np.where(span>0,span,1.)
        np.savez_compressed(str(RAW/'actual_keras_predictions.npz'),probabilities=model.predict(normalized,batch_size=1024))
        return
    from experiments.four_method_comparison.scripts import methods as m
    if a.action=='extract':
        chunks=[];labels=[]
        for case in ('sod','lax','shu_osher','burgers_post','advection_square'):
            with np.load(RAW/f'run_{case}_128_nn_teno5.npz') as z:
                for index in (0,len(z['states'])//2,len(z['states'])-1):
                    u=m.jnp.asarray(z['states'][index])
                    if u.ndim==2:s=m.euler_split_flux_stencils(u,'transmissive')
                    else:s=(m.gather(u,(-2,-1,0,1,2),'periodic'),m.gather(u,(3,2,1,0,-1),'periodic'))
                    for direction,v in enumerate(s):
                        chunks.append(np.asarray(v).reshape(-1,5));labels.extend([f'{case}:snapshot{index}:direction{direction}']*chunks[-1].shape[0])
        np.savez_compressed(RAW/'actual_classifier_inputs.npz',raw=np.concatenate(chunks),labels=np.asarray(labels))
    else:
        with np.load(RAW/'actual_classifier_inputs.npz') as z:raw=z['raw'];labels=z['labels']
        with np.load(RAW/'actual_keras_predictions.npz') as z:reference=z['probabilities']
        actual=np.asarray(m.original_probabilities(m.jnp.asarray(raw)));thresholds=np.asarray(m.THRESHOLDS)
        mismatch=np.any((actual<thresholds)!=(reference<thresholds),axis=-1)
        result=dict(stencil_count=len(raw),maximum_probability_difference=float(np.max(abs(actual-reference))),
                    mask_mismatch_count=int(mismatch.sum()),mismatch_sources=np.unique(labels[mismatch]).tolist(),
                    probability_tolerance=1e-6,thresholds=thresholds.tolist())
        (RAW/'actual_input_parity.json').write_text(json.dumps(result,indent=2))
        assert np.allclose(actual,reference,atol=1e-6,rtol=2e-5),result
        assert not mismatch.any(),result
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
