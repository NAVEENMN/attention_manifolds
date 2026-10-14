"""Initialize coordinate-wise Q/K spline curves from a pretrained Llama and audit parity."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from spline_qk import collect_domains, install, projections, save_splines


def blocks(text,tokenizer,length,count):
    tokens=tokenizer(text,add_special_tokens=False)['input_ids']
    n=min(len(tokens)//length,count)
    if n<1:raise ValueError('Not enough text for one block')
    return torch.tensor(tokens[:n*length],dtype=torch.long).view(n,length)


@torch.no_grad()
def evaluate(model,data,device,probe_positions,reference=None):
    total=0.;count=0;probes=[];max_error=0.;kl_sum=0.;kl_count=0;argmax_changed=0
    for i,batch in enumerate(data):
        ids=batch[None].to(device)
        out=model(input_ids=ids,labels=ids,use_cache=False)
        total+=out.loss.item()*(ids.shape[1]-1);count+=ids.shape[1]-1
        positions=torch.linspace(0,ids.shape[1]-1,min(probe_positions,ids.shape[1])).long()
        sample=out.logits[0,positions.to(device)].detach().cpu()
        if reference is None:probes.append(sample)
        else:
            base=reference[i].double();edited=sample.double()
            max_error=max(max_error,(base-edited).abs().max().item())
            lp,lq=base.log_softmax(-1),edited.log_softmax(-1)
            kl_sum+=(lp.exp()*(lp-lq)).sum(-1).clamp_min(0).sum().item()
            kl_count+=len(sample)
            argmax_changed+=(base.argmax(-1)!=edited.argmax(-1)).sum().item()
    nll=total/count
    metrics={'nll':nll,'ppl':math.exp(nll),'target_tokens':count}
    if reference is not None:
        metrics.update(sampled_logit_max_abs_error=max_error,sampled_mean_kl=kl_sum/kl_count,
                       sampled_argmax_changes=argmax_changed,sampled_positions=kl_count)
    return metrics,probes


class ProjectionAudit:
    def __init__(self,model,originals):
        self.stats={};self.handles=[]
        for key,module in projections(model).items():
            self.handles.append(module.register_forward_hook(self.make_hook(key,originals[key])))

    def make_hook(self,key,linear):
        @torch.no_grad()
        def hook(module,args,output):
            x=args[0];expected=F.linear(x,linear.weight,linear.bias)
            error=output-expected
            s=self.stats.setdefault(f'{key[0]}.{key[1]}',{'sse':0.,'reference_ss':0.,'max_abs_error':0.,'elements':0,'tail_inputs':0,'inputs':0})
            s['sse']+=error.double().square().sum().item()
            s['reference_ss']+=expected.double().square().sum().item()
            s['max_abs_error']=max(s['max_abs_error'],error.abs().max().item())
            s['elements']+=error.numel()
            s['tail_inputs']+=((expected-module.center).abs()>module.radius).sum().item();s['inputs']+=expected.numel()
        return hook

    def finish(self):
        for handle in self.handles:handle.remove()
        for s in self.stats.values():
            s['rmse']=math.sqrt(s['sse']/s['elements'])
            s['relative_rmse']=math.sqrt(s['sse']/max(s['reference_ss'],1e-30))
            s['linear_extrapolation_fraction']=s['tail_inputs']/s['inputs']
        return self.stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='meta-llama/Llama-3.2-1B-Instruct')
    p.add_argument('--revision',default='main')
    p.add_argument('--output',required=True,help='New run directory')
    p.add_argument('--calibration-text',help='UTF-8 training text; default WikiText-2 train')
    p.add_argument('--evaluation-text',help='Separate UTF-8 text; default WikiText-2 validation')
    p.add_argument('--n-basis',type=int,default=16)
    p.add_argument('--block-size',type=int,default=128)
    p.add_argument('--calibration-blocks',type=int,default=32)
    p.add_argument('--evaluation-blocks',type=int,default=16)
    p.add_argument('--probe-positions',type=int,default=8)
    p.add_argument('--domain-std',type=float,default=3.)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-relative-qk-error',type=float,default=1e-4)
    p.add_argument('--max-logit-error',type=float,default=.002)
    p.add_argument('--max-nll-change',type=float,default=.001)
    args=p.parse_args()
    if args.n_basis<4 or args.block_size<2 or min(args.calibration_blocks,args.evaluation_blocks,args.probe_positions)<1:
        p.error('Invalid basis/block counts')
    if not all(math.isfinite(v) and v>0 for v in [args.domain_std,args.max_relative_qk_error,args.max_logit_error,args.max_nll_change]):
        p.error('Domain and parity limits must be positive and finite')
    if bool(args.calibration_text)!=bool(args.evaluation_text):
        p.error('Supply both text files, or neither for WikiText-2')
    torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
    if torch.cuda.is_available():torch.backends.cuda.matmul.allow_tf32=False
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,dtype=torch.float32,attn_implementation='eager').to(args.device).eval()
    model.requires_grad_(False)
    if model.config.model_type!='llama':raise ValueError('Llama only')
    if args.block_size>model.config.max_position_embeddings:raise ValueError('Block exceeds context length')
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    if args.calibration_text:
        calibration_text=Path(args.calibration_text).read_text();evaluation_text=Path(args.evaluation_text).read_text()
        source={'calibration':args.calibration_text,'evaluation':args.evaluation_text}
    else:
        from datasets import load_dataset
        calibration_text='\n\n'.join(load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')['text'])
        evaluation_text='\n\n'.join(load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='validation')['text'])
        source={'dataset':'Salesforce/wikitext','configuration':'wikitext-2-raw-v1','calibration':'train','evaluation':'validation'}
    if calibration_text==evaluation_text:raise ValueError('Use separate calibration and evaluation text')
    calibration=blocks(calibration_text,tokenizer,args.block_size,args.calibration_blocks)
    evaluation=blocks(evaluation_text,tokenizer,args.block_size,args.evaluation_blocks)
    old_count=sum(m.weight.numel() for m in projections(model).values())
    curve_count=sum(m.out_features for m in projections(model).values())*args.n_basis
    print(json.dumps({'retained_qk_weights':old_count,'spline_coefficients':curve_count,
                      'spline_coefficient_GiB_float32':curve_count*4/2**30}),flush=True)
    baseline,reference=evaluate(model,evaluation,args.device,args.probe_positions)
    print('Original validation:',json.dumps(baseline),flush=True)
    domains=collect_domains(model,(b[None].to(args.device) for b in calibration),args.domain_std)
    originals=install(model,domains,args.n_basis)
    audit=ProjectionAudit(model,originals)
    try:
        calibrated,_=evaluate(model,evaluation,args.device,args.probe_positions,reference)
    finally:
        projection_metrics=audit.finish()
    relative=max(s['relative_rmse'] for s in projection_metrics.values())
    passed=(relative<=args.max_relative_qk_error and calibrated['sampled_logit_max_abs_error']<=args.max_logit_error
            and abs(calibrated['nll']-baseline['nll'])<=args.max_nll_change)
    metrics={'model':args.model,'revision':args.revision,'resolved_commit':getattr(model.config,'_commit_hash',None),
             'torch':str(torch.__version__),'transformers':transformers.__version__,'args':vars(args),
             'data_source':source,'calibration_sha256':hashlib.sha256(calibration_text.encode()).hexdigest(),
             'evaluation_sha256':hashlib.sha256(evaluation_text.encode()).hexdigest(),
             'calibration_blocks_used':len(calibration),'evaluation_blocks_used':len(evaluation),
             'baseline':baseline,'calibrated':calibrated,'projection_metrics':projection_metrics,
             'max_projection_relative_rmse':relative,'parity_checks_passed':passed,
             'initialization':'Coordinate identity curves after retained linear projections; no optimization',
             'coefficient_count':curve_count}
    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'calibrated':calibrated,'max_projection_relative_rmse':relative,'passed':passed}),flush=True)
    if not passed:raise RuntimeError('Parity limits exceeded; inspect metrics.json. No checkpoint saved.')
    del originals,reference,audit
    save_splines(directory/'checkpoint',model,metrics)
    print(f'Saved all Q/K spline projections to {directory / "checkpoint"}',flush=True)


if __name__=='__main__':main()
