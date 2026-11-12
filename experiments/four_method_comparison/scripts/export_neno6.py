"""Load author state_dict safely and evaluate author classes without running GUI code."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/four_method_comparison'
VENDOR = OUT / 'vendor/NNENO'

def author_class(filename, name):
    tree = ast.parse((VENDOR / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    ns = dict(torch=torch, nn=nn, np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), ns)
    return ns[name]

if __name__ == '__main__':
    torch.set_num_threads(1)
    Model = author_class('test_teno_interg_adv_function.py', 'Model')
    Scheme = author_class('globalfunction.py', 'Scheme6')
    weights = torch.load(VENDOR / 'mytrainmodel1', map_location='cpu', weights_only=True)
    model = Model(6, 4)
    model.load_state_dict(weights, strict=True)
    model.eval()
    rng = np.random.default_rng(81276)
    raw = np.vstack([rng.normal(size=(4096, 6)), np.zeros((1, 6)), np.full((1, 6), 7.)])
    raw = torch.tensor(raw, dtype=torch.float64)
    def evaluate(s):
        norm = s / torch.clamp(torch.max(torch.abs(s), dim=-1, keepdim=True).values, min=1.)
        with torch.no_grad():
            p = model(norm)
            f = Scheme().TENO6NN(s, p)
        return p, f
    p, f = evaluate(raw)
    n = 64; dx = 2. / n; dt = .2 * dx
    x = -1 + (np.arange(n) + .5) * dx
    initial = torch.tensor(np.sin(np.pi*x), dtype=torch.float64)
    u = initial.clone()
    for _ in range(8):
        def rhs(v):
            s = Scheme().inputValueToTENO6StencilsL(v)
            _, flux = evaluate(s)
            return -(flux - torch.roll(flux, 1)) / dx
        a = u + dt * rhs(u)
        b = .75*u + .25*(a + dt*rhs(a))
        u = u/3 + 2./3*(b + dt*rhs(b))
    arrays = {}
    for i in range(1, 6):
        arrays['w%d' % (i-1)] = weights['fc%d.weight' % i].numpy().T
        arrays['b%d' % (i-1)] = weights['fc%d.bias' % i].numpy()
    np.savez_compressed(OUT / 'results/raw/neno6_author_export.npz', raw=raw.numpy(),
                        probabilities=p.numpy(), flux=f.numpy(), smoke_initial=initial.numpy(),
                        left_stencils=Scheme().inputValueToTENO6StencilsL(initial).numpy(),
                        right_stencils=Scheme().inputValueToTENO6StencilsR(initial).numpy(),
                        smoke_final=u.numpy(), smoke_dt=dt, smoke_dx=dx, **arrays)
    audit = dict(source=str(VENDOR.relative_to(ROOT)), torch_version=torch.__version__,
                 weights_only=True, checkpoint_sha256=hashlib.sha256((VENDOR/'mytrainmodel1').read_bytes()).hexdigest(),
                 architecture=[6,64,32,16,8,4], parameters=sum(p.numel() for p in model.parameters()),
                 normalization='stencil / max(1, max(abs(stencil)))', troubled_threshold=.5,
                 all_discard_fixture_count=int(torch.sum(torch.all(p >= .5, dim=-1))),
                 smoke_finite=bool(torch.all(torch.isfinite(u))), smoke_steps=8, smoke_time=8*dt,
                 smoke_l1=float(torch.mean(torch.abs(u-torch.sin(torch.tensor(np.pi*(x-8*dt)))))))
    (OUT / 'results/raw/neno6_load_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
