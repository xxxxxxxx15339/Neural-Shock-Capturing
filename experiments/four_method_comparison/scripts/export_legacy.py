"""Read original Keras artifacts in the legacy environment; never train/save models."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
from pathlib import Path
import json
import hashlib
import numpy as np
from keras.models import load_model

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/four_method_comparison/results/raw'
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for path in sorted((ROOT / 'models').glob('*.h5')):
    try:
        model = load_model(str(path), compile=False)
        row = dict(path=str(path.relative_to(ROOT)), loaded=True,
                   input_shape=list(model.input_shape), output_shape=list(model.output_shape),
                   parameters=int(model.count_params()),
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if path.name == 'nn_teno5.h5':
            rng = np.random.RandomState(81276)
            raw = np.concatenate([rng.normal(size=(4096, 5)),
                                  np.full((1, 5), 0.), np.full((1, 5), 7.)])
            span = np.ptp(raw, axis=-1, keepdims=True)
            norm = (raw - raw.min(axis=-1, keepdims=True)) / np.where(span > 0, span, 1.)
            predictions = model.predict(norm, batch_size=1024)
            weights = model.get_weights()
            np.savez_compressed(str(OUT / 'original_keras_export.npz'),
                                raw=raw, probabilities=predictions,
                                **{'a%d' % i: a for i, a in enumerate(weights)})
    except Exception as exc:
        row = dict(path=str(path.relative_to(ROOT)), loaded=False, error=repr(exc))
    rows.append(row)
(OUT / 'keras_load_audit.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
