"""Frozen linear Q/K projections followed by trainable coordinate-wise curves.

z = W x + b; y[o] = sum_m C[o,m] B_m(z[o]).
One spline per projected coordinate, not one per matrix connection.
"""
import json
from pathlib import Path
import torch
from torch import nn

FORMAT='qk-coordinate-curves-v1'


def basis_and_derivative(t,n_basis):
    u=t.unsqueeze(-1)*(n_basis-3)-torch.arange(n_basis,device=t.device,dtype=t.dtype)+3
    z=u.clamp(0,4);y=z-1;v=3-z
    b=torch.where(z<1,z**3/6,torch.where(z<2,(-3*y**3+3*y*y+3*y+1)/6,
        torch.where(z<3,(-3*v**3+3*v*v+3*v+1)/6,(4-z)**3/6)))
    db=torch.where(z<1,z*z/2,torch.where(z<2,(-9*y*y+6*y+3)/6,
        torch.where(z<3,(9*v*v-6*v-3)/6,-(4-z)**2/2)))*(n_basis-3)
    valid=(u>=0)&(u<=4)
    return torch.where(valid,b,0.),torch.where(valid,db,0.)


class SplineProjection(nn.Module):
    def __init__(self,linear,n_basis=16):
        super().__init__()
        if not isinstance(linear,nn.Linear):raise TypeError('Expected original nn.Linear')
        if n_basis<4:raise ValueError('At least four basis coefficients required')
        if linear.weight.dtype not in (torch.float32,torch.float64):raise ValueError('Float32 required')
        self.linear=linear.requires_grad_(False)
        self.in_features,self.out_features,self.n_basis=linear.in_features,linear.out_features,n_basis
        device,dtype=linear.weight.device,linear.weight.dtype
        self.coefficients=nn.Parameter(torch.zeros(self.out_features,n_basis,device=device,dtype=dtype))
        self.register_buffer('center',torch.zeros(self.out_features,device=device,dtype=dtype))
        self.register_buffer('radius',torch.ones(self.out_features,device=device,dtype=dtype))

    def features(self,z):
        t=.5+(z-self.center)/(2*self.radius);tc=t.clamp(0,1)
        b,db=basis_and_derivative(tc,self.n_basis)
        return b+(t-tc).unsqueeze(-1)*db

    def forward(self,x):
        if x.dtype!=self.coefficients.dtype or x.dtype not in (torch.float32,torch.float64):
            raise ValueError('Float32 required; mixed precision is unsupported')
        z=self.linear(x)
        return (self.features(z)*self.coefficients).sum(-1)

    @classmethod
    @torch.no_grad()
    def from_linear(cls,linear,center,radius,n_basis=16):
        result=cls(linear,n_basis)
        center=torch.as_tensor(center,device=linear.weight.device,dtype=linear.weight.dtype)
        radius=torch.as_tensor(radius,device=linear.weight.device,dtype=linear.weight.dtype)
        if center.shape!=(linear.out_features,) or radius.shape!=center.shape:
            raise ValueError('Domains must have one entry per projected coordinate')
        if not torch.isfinite(center).all() or not torch.isfinite(radius).all() or (radius<=0).any():
            raise ValueError('Finite centers and positive radii required')
        result.center.copy_(center);result.radius.copy_(radius)
        g=center[:,None]-radius[:,None]+2*radius[:,None]*(
            torch.arange(n_basis,device=center.device,dtype=center.dtype)-1)/(n_basis-3)
        result.coefficients.copy_(g)
        return result

    def curve_state(self):
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if not k.startswith('linear.')}


def projections(model):
    return {(i,kind):getattr(layer.self_attn,kind+'_proj')
            for i,layer in enumerate(model.model.layers) for kind in ('q','k')}


def install(model,domains,n_basis=16):
    if model.config.model_type!='llama':raise ValueError('HF Llama only')
    original=projections(model)
    if set(domains)!=set(original):raise ValueError('Domains required for every Q/K projection')
    if any(not isinstance(m,nn.Linear) for m in original.values()):raise ValueError('Use the original backbone')
    model.requires_grad_(False)
    for (i,kind),linear in original.items():
        d=domains[i,kind]
        setattr(model.model.layers[i].self_attn,kind+'_proj',
                SplineProjection.from_linear(linear,d['center'],d['radius'],n_basis))
    return original


@torch.no_grad()
def collect_domains(model,batches,std_multiplier=3.):
    """Domains for raw projected Q/K values, before RoPE; separate per coordinate."""
    if std_multiplier<=0:raise ValueError('Positive domain multiplier required')
    stats={};hooks=[]
    def make_hook(key):
        def hook(module,args,output):
            z=output.detach().reshape(-1,module.out_features)
            s=z.sum(0,dtype=torch.float64);ss=z.double().square().sum(0)
            if key not in stats:stats[key]=[s,ss,len(z)]
            else:stats[key][0]+=s;stats[key][1]+=ss;stats[key][2]+=len(z)
        return hook
    was_training=model.training;model.eval()
    try:
        for key,linear in projections(model).items():hooks.append(linear.register_forward_hook(make_hook(key)))
        for ids in batches:model(input_ids=ids,use_cache=False)
    finally:
        for h in hooks:h.remove()
        model.train(was_training)
    if len(stats)!=len(projections(model)):raise ValueError('No complete calibration data')
    domains={}
    for key,(s,ss,n) in stats.items():
        mean=s/n;std=(ss/n-mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
        domains[key]={'center':mean.float().cpu(),'radius':(std_multiplier*std).float().cpu(),'tokens':n}
    return domains


def save_splines(directory,model,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    entries=[]
    for i,layer in enumerate(model.model.layers):
        pair={kind:getattr(layer.self_attn,kind+'_proj') for kind in ('q','k')}
        if not all(isinstance(m,SplineProjection) for m in pair.values()):raise ValueError('All Q/K need curves')
        filename=f'layer_{i:03d}.pt'
        torch.save({kind:m.curve_state() for kind,m in pair.items()},directory/filename)
        entries.append({'layer':i,'file':filename,'n_basis':pair['q'].n_basis})
    (directory/'manifest.json').write_text(json.dumps({'format':FORMAT,'layers':entries,'metadata':metadata},indent=2)+'\n')


def load_splines(directory,model):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT:raise ValueError('Incompatible checkpoint')
    if model.config.model_type!='llama':raise ValueError('HF Llama only')
    if [e['layer'] for e in manifest['layers']]!=list(range(len(model.model.layers))):
        raise ValueError('Checkpoint layer mismatch')
    model.requires_grad_(False)
    for entry in manifest['layers']:
        i=entry['layer'];states=torch.load(directory/entry['file'],map_location='cpu',weights_only=True)
        for kind,state in states.items():
            old=getattr(model.model.layers[i].self_attn,kind+'_proj')
            new=SplineProjection.from_linear(old,state['center'],state['radius'],entry['n_basis'])
            if set(state)!={'coefficients','center','radius'}:raise ValueError('Invalid curve checkpoint keys')
            with torch.no_grad():new.coefficients.copy_(state['coefficients'])
            setattr(model.model.layers[i].self_attn,kind+'_proj',new)
    return manifest['metadata']
