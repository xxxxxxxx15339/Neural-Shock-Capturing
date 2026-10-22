#!/usr/bin/env python3
"""Generate a reproducible, analytic, cell-average NN-TENO5 dataset.

Ground truth is MATHEMATICAL, not physical measurements. Includes analytic
periodic scalar transport, pre-shock smooth Burgers, entropy Burgers top-hat,
and exact gamma=1.4 Euler Riemann solutions. Never substitutes point samples
for cell averages. Validation/test are whole independent trajectories.

pip install numpy scipy h5py
python build_dataset.py --output DATA_DIR --scalar 4096 --euler 2048
"""
import argparse, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import h5py
from scipy.special import erf
from scipy.optimize import brentq

GAMMA = 1.4
GL16 = np.polynomial.legendre.leggauss(16)
GL32 = np.polynomial.legendre.leggauss(32)
SPLITS = ('train','validation','test')
FAMILIES = {
 'scalar': {'fourier':.22,'square':.22,'gaussian':.11,'burgers_smooth':.23,'burgers_top_hat':.17,'constant':.05},
 'euler': {'random_riemann':.65,'sod_like':.15,'contact':.10,'two_rarefactions':.05,'constant':.05},
}
NS = [64,128,256]
RATIO=(.70,.15,.15)

def alloc(n, fractions):
    a=np.array(list(fractions),float); a=a/a.sum()
    k=np.floor(a*n).astype(int)
    for j in np.argsort(-(a*n-k))[:n-k.sum()]:k[j]+=1
    return k

def average_samples(func,n,breaks=(),gl=GL16):
    """Adaptive-to-known-wave-boundaries Gauss integration; all cells exact to quadrature accuracy."""
    z,w=gl; edges=np.arange(n+1,dtype=float)/n
    out=None
    for i in range(n):
        left,right=edges[i],edges[i+1]
        xs=[left]+[float(b) for b in breaks if left+1e-15<b<right-1e-15]+[right]
        acc=None
        for a,b in zip(xs[:-1],xs[1:]):
            x=a+0.5*(z+1)*(b-a)
            v=np.asarray(func(x),float)
            piece=0.5*(b-a)*np.tensordot(w,v,axes=(0,0))
            acc=piece if acc is None else acc+piece
        if out is None:out=np.zeros((n,)+np.shape(acc))
        out[i]=n*acc
    return out

def exact_square(n,left,right,inside,outside,shift):
    xlo=np.arange(n)/n; xhi=(np.arange(n)+1)/n
    c=np.zeros(n)
    for k in range(-2,4):
        a,b=left+shift+k,right+shift+k
        c+=n*np.maximum(0,np.minimum(xhi,b)-np.maximum(xlo,a))
    return outside+(inside-outside)*c

def exact_fourier(n,offset,terms,t,speed):
    dx=1/n; x=(np.arange(n)+0.5)*dx
    u=np.full(n,offset)
    for amp,k,phase in terms:u+=amp*np.sinc(k*dx)*np.sin(2*np.pi*k*(x-speed*t)+phase)
    return u

def exact_gaussian(n,mu,sigma,amp,offset,t,speed):
    xl=np.arange(n)/n; xr=(np.arange(n)+1)/n
    total=np.full(n,offset)
    b=sigma*np.sqrt(2.0)
    for k in range(-3,4):
        center=mu+speed*t+k
        total+=amp*sigma*np.sqrt(np.pi/2)*n*(erf((xr-center)/b)-erf((xl-center)/b))
    return total

def exact_burgers_smooth(n,offset,amp,k,phase,t,gl=GL32):
    def values(x):
        if t==0:return offset+amp*np.sin(2*np.pi*k*x+phase)
        v=offset+amp*np.sin(2*np.pi*k*(x-offset*t)+phase)
        for it in range(22):
            angle=2*np.pi*k*(x-t*v)+phase
            d=v-offset-amp*np.sin(angle)
            j=1+amp*2*np.pi*k*t*np.cos(angle)
            v-=d/j
        residual=np.max(np.abs(v-offset-amp*np.sin(2*np.pi*k*(x-t*v)+phase)))
        if residual>2e-12:raise ValueError(f'Burgers characteristic residual {residual}')
        return v
    z,w=gl
    x=(np.arange(n)[:,None]+0.5*(z[None,:]+1.0))/n
    v=values(x)
    return 0.5*(v*w[None,:]).sum(axis=1)

def exact_burgers_hat(n,lo,hi,left,right,t):
    if t==0:return exact_square(n,left,right,hi,lo,0.)
    A=left+lo*t; B=left+hi*t; C=right+0.5*(lo+hi)*t
    if B>=C:raise ValueError('Burgers wave interaction outside exact formula')
    cell_l=np.arange(n)/n; cell_r=(np.arange(n)+1)/n
    result=np.full(n,lo)
    a=np.maximum(cell_l,A); b=np.minimum(cell_r,B); span=np.maximum(0,b-a)
    # integrate (x-left)/t - lo over the rarefaction portion
    result+=n*np.where(span>0,((b-left)**2-(a-left)**2)/(2*t)-lo*(b-a),0.)
    result+=(hi-lo)*n*np.maximum(0,np.minimum(cell_r,C)-np.maximum(cell_l,B))
    return result

def scalar_case(rng,n,family):
    t_end=float(rng.uniform(.08,.16)); times=np.array([0.,t_end/3,2*t_end/3,t_end],float)
    if family=='constant':
        u=float(rng.uniform(-.5,1.5)); meta={'value':u}
        state=np.tile(np.full(n,u),(4,1))
    elif family=='square':
        left=float(rng.uniform(.10,.30)); right=float(rng.uniform(.55,.82));
        low=float(rng.uniform(-.4,.4)); high=float(low+rng.uniform(.3,1.5));speed=float(rng.uniform(.5,1.4))
        meta={'left':left,'right':right,'low':low,'high':high,'speed':speed}
        state=np.array([exact_square(n,left,right,high,low,speed*t) for t in times])
    elif family=='fourier':
        offset=float(rng.uniform(-.4,.8)); speed=float(rng.uniform(.5,1.4))
        terms=[[float(rng.uniform(-.5,.5)),int(kk),float(rng.uniform(0,2*np.pi))] for kk in rng.choice(np.arange(1,7),size=int(rng.integers(1,4)),replace=False)]
        meta={'offset':offset,'speed':speed,'terms':terms}
        state=np.array([exact_fourier(n,offset,terms,t,speed) for t in times])
    elif family=='gaussian':
        mu=float(rng.uniform(.15,.85));sig=float(rng.uniform(.025,.11));amp=float(rng.uniform(.15,1.2));off=float(rng.uniform(-.3,.7));speed=float(rng.uniform(.5,1.4))
        meta={'mu':mu,'sigma':sig,'amplitude':amp,'offset':off,'speed':speed}
        state=np.array([exact_gaussian(n,mu,sig,amp,off,t,speed) for t in times])
    elif family=='burgers_smooth':
        off=float(rng.uniform(.1,.85));amp=float(rng.uniform(.035,.16));k=int(rng.integers(1,4));phase=float(rng.uniform(0,2*np.pi))
        ts=1/(2*np.pi*k*amp);t_end=min(t_end,.58*ts);times=np.array([0.,t_end/3,2*t_end/3,t_end])
        meta={'offset':off,'amplitude':amp,'wavenumber':k,'phase':phase,'shock_time':ts}
        state=np.array([exact_burgers_smooth(n,off,amp,k,phase,t) for t in times])
    elif family=='burgers_top_hat':
        lo=float(rng.uniform(.05,.35));hi=float(lo+rng.uniform(.25,.85));left=float(rng.uniform(.14,.30));right=float(rng.uniform(.65,.81))
        # ensure waves cannot reach periodic boundaries or one another
        t_end=min(t_end,.6*(right-left)/(hi-lo),.75*(1-right)/hi)
        times=np.array([0.,t_end/3,2*t_end/3,t_end]);meta={'low':lo,'high':hi,'left':left,'right':right,'max_time_before_interaction':2*(right-left)/(hi-lo)}
        state=np.array([exact_burgers_hat(n,lo,hi,left,right,t) for t in times])
    else:raise ValueError(family)
    return times,state,meta

def primitive_to_u(rho,u,p):return np.array([rho,rho*u,p/(GAMMA-1)+.5*rho*u*u])
def f_side(p,rho,u,pk):
    c=math.sqrt(GAMMA*pk/rho)
    if p>pk:
        A=2/((GAMMA+1)*rho);B=pk*(GAMMA-1)/(GAMMA+1)
        return (p-pk)*math.sqrt(A/(p+B))
    return 2*c/(GAMMA-1)*((p/pk)**((GAMMA-1)/(2*GAMMA))-1)

class ExactEuler:
    def __init__(self,lp,rp,x0):
        self.lp,self.rp,self.x0=tuple(lp),tuple(rp),float(x0)
        rl,ul,pl=lp;rr,ur,pr=rp
        cl=math.sqrt(GAMMA*pl/rl);cr=math.sqrt(GAMMA*pr/rr)
        if ur-ul>=2*(cl+cr)/(GAMMA-1):raise ValueError('vacuum Riemann problem')
        def f(p):return f_side(p,rl,ul,pl)+f_side(p,rr,ur,pr)+(ur-ul)
        upper=max(pl,pr,1.)
        while f(upper)<0:upper*=2
        star=brentq(f,1e-14,upper,xtol=2e-14,rtol=1e-14)
        vel=.5*(ul+ur+f_side(star,rr,ur,pr)-f_side(star,rl,ul,pl))
        eta=(GAMMA-1)/(GAMMA+1)
        def sr(rho,pk):
            return rho*((star/pk+eta)/(eta*(star/pk)+1)) if star>pk else rho*(star/pk)**(1/GAMMA)
        rsl,rsr=sr(rl,pl),sr(rr,pr)
        self.star,self.vel,self.rsl,self.rsr=star,vel,rsl,rsr
        self.left_shock,self.right_shock=star>pl,star>pr
        if self.left_shock:
            sl=ul-cl*math.sqrt((GAMMA+1)*star/(2*GAMMA*pl)+(GAMMA-1)/(2*GAMMA)); self.L=(sl,sl)
        else:self.L=(ul-cl,vel-math.sqrt(GAMMA*star/rsl))
        if self.right_shock:
            sr=ur+cr*math.sqrt((GAMMA+1)*star/(2*GAMMA*pr)+(GAMMA-1)/(2*GAMMA)); self.R=(sr,sr)
        else:self.R=(vel+math.sqrt(GAMMA*star/rsr),ur+cr)
        self.all_speeds=(self.L[0],self.L[1],vel,self.R[0],self.R[1])
        self.max_speed=max(abs(v) for v in self.all_speeds)
        self.types=('shock' if self.left_shock else 'rarefaction','contact','shock' if self.right_shock else 'rarefaction')
        self.F_L=self.flux(*lp);self.F_R=self.flux(*rp)
    @staticmethod
    def flux(rho,u,p):return np.array([rho*u,rho*u*u+p,u*(p/(GAMMA-1)+.5*rho*u*u+p)])
    def __call__(self,x,t):
        x=np.asarray(x,dtype=float);xl,ul,pl=self.lp;xr,ur,pr=self.rp
        rho=np.empty_like(x);vel=np.empty_like(x);pressure=np.empty_like(x)
        if t==0:
            left=x<self.x0;rho[:]=np.where(left,xl,xr);vel[:]=np.where(left,ul,ur);pressure[:]=np.where(left,pl,pr)
        else:
            xi=(x-self.x0)/t; left=xi<=self.vel
            if self.left_shock:
                orig=left&(xi<self.L[0]); middle=left&~orig
                rho[orig],vel[orig],pressure[orig]=xl,ul,pl
                rho[middle],vel[middle],pressure[middle]=self.rsl,self.vel,self.star
            else:
                orig=left&(xi<self.L[0]);middle=left&(xi>=self.L[1]);fan=left&~(orig|middle)
                rho[orig],vel[orig],pressure[orig]=xl,ul,pl
                rho[middle],vel[middle],pressure[middle]=self.rsl,self.vel,self.star
                vf=2/(GAMMA+1)*(math.sqrt(GAMMA*pl/xl)+.5*(GAMMA-1)*ul+xi[fan]);cf=2/(GAMMA+1)*(math.sqrt(GAMMA*pl/xl)+.5*(GAMMA-1)*(ul-xi[fan]))
                rho[fan]=xl*(cf/math.sqrt(GAMMA*pl/xl))**(2/(GAMMA-1));vel[fan]=vf;pressure[fan]=pl*(rho[fan]/xl)**GAMMA
            right=~left
            if self.right_shock:
                orig=right&(xi>self.R[1]);middle=right&~orig
                rho[orig],vel[orig],pressure[orig]=xr,ur,pr
                rho[middle],vel[middle],pressure[middle]=self.rsr,self.vel,self.star
            else:
                orig=right&(xi>self.R[1]);middle=right&(xi<=self.R[0]);fan=right&~(orig|middle)
                rho[orig],vel[orig],pressure[orig]=xr,ur,pr
                rho[middle],vel[middle],pressure[middle]=self.rsr,self.vel,self.star
                cf=2/(GAMMA+1)*(math.sqrt(GAMMA*pr/xr)-.5*(GAMMA-1)*(ur-xi[fan]));vf=2/(GAMMA+1)*(-math.sqrt(GAMMA*pr/xr)+.5*(GAMMA-1)*ur+xi[fan])
                rho[fan]=xr*(cf/math.sqrt(GAMMA*pr/xr))**(2/(GAMMA-1));vel[fan]=vf;pressure[fan]=pr*(rho[fan]/xr)**GAMMA
        return np.stack((rho,rho*vel,pressure/(GAMMA-1)+.5*rho*vel**2),axis=-1)
    def averages(self,n,t,gl=GL16):
        # Vectorized integration: split exactly at ALL wave boundaries, then
        # apply Gaussian quadrature within each smooth/constant interval.
        z,w=gl
        wave_x=[self.x0+v*t for v in self.all_speeds] if t>0 else [self.x0]
        edges=np.array([0.,1.]+[float(q) for q in wave_x if 0.<q<1.])
        edges=np.unique(np.sort(edges))
        left=np.arange(n)[:,None]/n
        right=(np.arange(n)[:,None]+1.)/n
        a=np.maximum(left,edges[:-1][None,:])
        b=np.minimum(right,edges[1:][None,:])
        width=np.maximum(0.,b-a)
        mid=0.5*(a+b)
        x=mid[:,:,None]+0.5*width[:,:,None]*z[None,None,:]
        val=self(x,t)      # (cells, wave intervals, quadrature nodes, 3)
        weighted=np.einsum('ikqc,q->ikc',val,w)
        return 0.5*n*np.sum(width[:,:,None]*weighted,axis=1)

def euler_case(rng,n,family):
    x0=float(rng.uniform(.38,.62))
    if family=='constant':
        rho=float(rng.uniform(.3,2.5));u=float(rng.uniform(-.5,.5));p=float(rng.uniform(.25,2.5))
        l=r=(rho,u,p)
    elif family=='contact':
        rho_l=float(rng.uniform(.3,2.5));rho_r=float(rng.uniform(.3,2.5));u=float(rng.uniform(-.5,.5));p=float(rng.uniform(.35,2.))
        l=(rho_l,u,p);r=(rho_r,u,p)
    elif family=='sod_like':
        a=float(rng.uniform(.7,1.8));b=float(rng.uniform(.07,.25));l=(float(rng.uniform(.85,1.6)),float(rng.uniform(-.2,.2)),a);r=(float(rng.uniform(.09,.35)),float(rng.uniform(-.2,.2)),b)
    elif family=='two_rarefactions':
        l=(float(rng.uniform(.8,1.8)),float(rng.uniform(-.7,-.2)),float(rng.uniform(.6,1.6)))
        r=(float(rng.uniform(.8,1.8)),float(rng.uniform(.2,.7)),float(rng.uniform(.6,1.6)))
    elif family=='random_riemann':
        l=(float(rng.uniform(.2,2.8)),float(rng.uniform(-.8,.8)),float(rng.uniform(.2,2.8)))
        r=(float(rng.uniform(.2,2.8)),float(rng.uniform(-.8,.8)),float(rng.uniform(.2,2.8)))
    else:raise ValueError(family)
    solver=ExactEuler(l,r,x0)
    # All Riemann waves remain inside [0,1]: exact entire-domain boundary budget.
    t_max=.66*min(x0,1-x0)/max(solver.max_speed,1e-9)
    end=min(float(rng.uniform(.085,.16)),t_max)
    times=np.array([0.,end/3,2*end/3,end])
    u=np.stack([solver.averages(n,t) for t in times])
    meta={'left_primitive':l,'right_primitive':r,'jump':x0,'gamma':GAMMA,'waves':solver.types,'wave_speeds':solver.all_speeds,'max_wave_speed':solver.max_speed,'initial_left_flux':solver.F_L.tolist(),'initial_right_flux':solver.F_R.tolist()}
    return times,u,meta

def short_horizon(physics,family,times,u,meta,n,steps=15,cfl=.35):
    """Return a CFL-bounded target exactly aligned with `steps` SSPRK3 steps.

    This avoids interpolating shocks between output snapshots. Time is chosen
    from a conservative estimated characteristic bound for each example.
    Euler bound includes Roe/fan speeds and all sampled characteristic speeds.
    """
    if physics=='euler':
        rho=u[...,0];v=u[...,1]/rho;p=(GAMMA-1)*(u[...,2]-.5*rho*v*v)
        speed_est=float(np.max(np.abs(v)+np.sqrt(GAMMA*p/rho)))
        speed_est=max(speed_est,float(meta['max_wave_speed']),1e-6)*1.2
    elif family.startswith('burgers_'):
        speed_est=max(float(np.max(np.abs(u))),1e-6)*1.2
    else:
        speed_est=abs(float(meta.get('speed',1.0)))*1.2
    time=min(.8*float(times[-1]),steps*cfl/(n*speed_est))
    dt=time/steps
    if physics=='euler':
        sol=ExactEuler(meta['left_primitive'],meta['right_primitive'],meta['jump'])
        target=sol.averages(n,time)
    elif family=='constant':target=np.array(u[0],copy=True)
    elif family=='square':target=exact_square(n,meta['left'],meta['right'],meta['high'],meta['low'],meta['speed']*time)
    elif family=='fourier':target=exact_fourier(n,meta['offset'],meta['terms'],time,meta['speed'])
    elif family=='gaussian':target=exact_gaussian(n,meta['mu'],meta['sigma'],meta['amplitude'],meta['offset'],time,meta['speed'])
    elif family=='burgers_smooth':target=exact_burgers_smooth(n,meta['offset'],meta['amplitude'],meta['wavenumber'],meta['phase'],time)
    elif family=='burgers_top_hat':target=exact_burgers_hat(n,meta['low'],meta['high'],meta['left'],meta['right'],time)
    else:raise ValueError(family)
    if not np.isfinite(target).all():raise ValueError('nonfinite short training target')
    return target,float(time),float(dt),float(speed_est)

def qa_case(physics,kind,times,u,meta,n,full_quad=False):
    assert u.shape==(4,n,3) if physics=='euler' else u.shape==(4,n)
    if not np.all(np.isfinite(u)):raise ValueError('non-finite arrays')
    if not np.all(np.diff(times)>0):raise ValueError('non-increasing times')
    if physics=='euler':
        rho=u[...,0];vel=u[...,1]/rho;press=(GAMMA-1)*(u[...,2]-.5*rho*vel**2)
        if np.min(rho)<=0 or np.min(press)<=0:raise ValueError('nonphysical averaged Euler state')
        mass=np.mean(u,axis=1)
        fl=np.array(meta['initial_left_flux']);fr=np.array(meta['initial_right_flux'])
        errors=np.max(np.abs(mass-(mass[0]-times[:,None]*(fr-fl)[None,:])),axis=0)
        tol=1e-8
        if np.max(errors)>tol:raise ValueError(f'failed Euler flux budget: {errors}')
        if full_quad and kind!='constant':
            slv=ExactEuler(meta['left_primitive'],meta['right_primitive'],meta['jump'])
            a=slv.averages(n,times[-1],gl=GL32)
            err=float(np.max(abs(a-u[-1])))
            if err>2e-10:raise ValueError(f'quad convergence Euler: {err}')
    else:
        means=np.mean(u,axis=1)
        err=float(np.max(np.abs(means-means[0])))
        if err>1e-9:raise ValueError(f'failed periodic scalar mass: {err}')
        errors=[err]
        if full_quad and kind=='burgers_smooth':
            p=meta;a=exact_burgers_smooth(n,p['offset'],p['amplitude'],p['wavenumber'],p['phase'],times[-1],GL16)
            err=float(np.max(np.abs(a-u[-1])))
            if err>2e-10:raise ValueError(f'Burgers quadrature: {err}')
    return float(np.max(errors))

def build(out,seed,scalar_count,euler_count):
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed)
    by_split={k:[] for k in SPLITS}
    manifests=[]; audit={'seed':seed,'generation':'exact analytic reference with Gauss-Legendre cell-average integration','physical_measurements_included':False,'counts':{},'max_conservation_residual':{},'qa_failures':[], 'split_disjoint':False}
    for physics,total in [('scalar',scalar_count),('euler',euler_count)]:
        split_counts=alloc(total,RATIO)
        for split,nsplit in zip(SPLITS,split_counts):
            family_names=list(FAMILIES[physics]);f_counts=alloc(int(nsplit),list(FAMILIES[physics].values()))
            jobs=[]
            for name,count in zip(family_names,f_counts):jobs.extend([name]*int(count))
            rng.shuffle(jobs)
            records=defaultdict(list)
            for k,kind in enumerate(jobs):
                n=int(NS[k%len(NS)])
                case_seed=int(rng.integers(1,2**63-1)); crng=np.random.default_rng(case_seed)
                try:
                    times,u,meta=(scalar_case if physics=='scalar' else euler_case)(crng,n,kind)
                    res=qa_case(physics,kind,times,u,meta,n,full_quad=(k%25==0))
                    short_target,short_time,short_dt,speed_bound=short_horizon(physics,kind,times,u,meta,n)
                except Exception as e:
                    audit['qa_failures'].append({'physics':physics,'split':split,'family':kind,'seed':case_seed,'error':str(e)})
                    raise
                key=f'{physics}/N{n:03d}'
                identifier=f'{physics}-{split}-{k:06d}-{case_seed}'
                data=dict(id=identifier,split=split,physics=physics,family=kind,cells=n,seed=case_seed,t_end=float(times[-1]),params=meta,source_class='mathematical_exact_solution',reference='cell_average',provenance='self-contained exact formulas / independent Riemann equations',qa_budget_max=res,short_time=short_time,short_dt=short_dt,short_steps=15,characteristic_speed_bound=speed_bound)
                records[key].append((times,u,short_target,data))
                manifests.append(data)
            h5=out/f'{physics}_{split}.h5'
            with h5py.File(h5,'w') as f:
                f.attrs['physics']=physics;f.attrs['split']=split;f.attrs['ground_truth']='analytic exact cell averages (not experimental measurements)'
                f.attrs['gamma']=GAMMA;f.attrs['seed']=seed
                for group,record in sorted(records.items()):
                    g=f.create_group(group.split('/')[1]);tm=np.stack([p[0] for p in record]);states=np.stack([p[1] for p in record]);shorts=np.stack([p[2] for p in record]);
                    g.create_dataset('times',data=tm,compression='gzip',compression_opts=4,shuffle=True)
                    g.create_dataset('states',data=states,compression='gzip',compression_opts=4,shuffle=True,chunks=(min(16,len(record)),)+states.shape[1:])
                    g.create_dataset('short_target',data=shorts,compression='gzip',compression_opts=4,shuffle=True)
                    g.create_dataset('short_time',data=np.array([p[3]['short_time'] for p in record]))
                    g.create_dataset('short_dt',data=np.array([p[3]['short_dt'] for p in record]))
                    g.create_dataset('advection_speed',data=np.array([p[3]['params'].get('speed',1.0) if physics=='scalar' and p[3]['family'] in ('constant','square','gaussian','fourier') else 0.0 for p in record]))
                    g.attrs['short_steps']=15
                    g.create_dataset('case_id',data=np.array([p[3]['id'] for p in record],dtype=h5py.string_dtype('utf-8')))
                    g.create_dataset('family',data=np.array([p[3]['family'] for p in record],dtype=h5py.string_dtype('utf-8')))
                    g.attrs['coordinates']='x_j=(j+0.5)/N; finite-volume averages over [j/N,(j+1)/N]'
            count=len(jobs);audit['counts'][f'{physics}_{split}']=count
            audit['max_conservation_residual'][f'{physics}_{split}']=max(d['qa_budget_max'] for d in manifests if d['physics']==physics and d['split']==split)
            print(physics,split,count,h5.stat().st_size//1024,'KiB',flush=True)
    if len({d['seed'] for d in manifests})!=len(manifests):raise ValueError('duplicate random problem seeds')
    (out/'cases.jsonl').write_text(''.join(json.dumps(m,sort_keys=True)+'\n' for m in manifests))
    audit['split_disjoint']=all(set(d['seed'] for d in manifests if d['split']==a).isdisjoint(set(d['seed'] for d in manifests if d['split']==b)) for a,b in [('train','validation'),('train','test'),('validation','test')])
    audit['family_counts']={f'{physics}_{sp}':dict(Counter(d['family'] for d in manifests if d['physics']==physics and d['split']==sp)) for physics in ('scalar','euler') for sp in SPLITS}
    audit['grid_counts']={f'{physics}_{sp}':dict(Counter(str(d['cells']) for d in manifests if d['physics']==physics and d['split']==sp)) for physics in ('scalar','euler') for sp in SPLITS}
    audit['manifest_sha256']=hashlib.sha256((out/'cases.jsonl').read_bytes()).hexdigest()
    files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob('*.h5'))}
    audit['files']=files; (out/'quality_report.json').write_text(json.dumps(audit,indent=2))
    print('TOTAL cases',len(manifests),'totalMB',sum(p.stat().st_size for p in out.glob('*.h5'))/1e6,flush=True)
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20261015);p.add_argument('--scalar',type=int,default=4096);p.add_argument('--euler',type=int,default=2048)
    z=p.parse_args();build(Path(z.output),z.seed,z.scalar,z.euler)
