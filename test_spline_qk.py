import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM
from spline_qk import SplineProjection,collect_domains,install,load_splines,projections,save_splines


def tiny():
    torch.manual_seed(10)
    c=LlamaConfig(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                  num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.)
    c._attn_implementation='eager'
    return LlamaForCausalLM(c).float().eval()


class ProjectionTests(unittest.TestCase):
    def test_linear_reproduction_inside_and_outside_domain(self):
        torch.manual_seed(2)
        for m in [4,8,16]:
            linear=nn.Linear(5,3,bias=True).double()
            center=torch.randn(3,dtype=torch.double);radius=torch.rand(3,dtype=torch.double)+.1
            spline=SplineProjection.from_linear(linear,center,radius,m)
            x=torch.cat([torch.randn(20,5,dtype=torch.double)*.05,torch.randn(20,5,dtype=torch.double)*20])
            torch.testing.assert_close(spline(x),linear(x),rtol=1e-11,atol=1e-11)

    def test_partition_and_linear_precision_with_extrapolation(self):
        spline=SplineProjection(nn.Linear(1,1).double(),8)
        x=torch.tensor([[-100.],[-1.],[-.3],[0.],[.8],[1.],[100.]],dtype=torch.double)
        b=spline.features(x)
        g=-1+2*(torch.arange(8,dtype=torch.double)-1)/5
        torch.testing.assert_close(b.sum(-1),torch.ones_like(x),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close((b*g).sum(-1),x,atol=1e-12,rtol=1e-12)

    def test_input_and_coefficient_gradients(self):
        linear=nn.Linear(3,2).double()
        spline=SplineProjection.from_linear(linear,torch.zeros(2),torch.ones(2))
        x=torch.tensor([[-3.,.2,4.]],dtype=torch.double,requires_grad=True)
        expected=torch.autograd.grad(linear(x).sum(),x)[0]
        spline(x).sum().backward()
        torch.testing.assert_close(x.grad,expected,rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.isfinite(spline.coefficients.grad).all())
        self.assertGreater(spline.coefficients.grad.abs().sum().item(),0)
        self.assertTrue(torch.autograd.gradcheck(lambda t:spline(t),(x,)))

    def test_nonlinear_learning_is_possible(self):
        linear=nn.Linear(1,1,bias=False)
        with torch.no_grad():linear.weight.fill_(1.)
        spline=SplineProjection.from_linear(linear,torch.zeros(1),torch.ones(1),8)
        x=torch.linspace(-1,1,64)[:,None];target=x.square()
        before=(spline(x)-target).square().mean().item()
        opt=torch.optim.Adam(spline.parameters(),lr=.05)
        for _ in range(100):
            opt.zero_grad();loss=(spline(x)-target).square().mean();loss.backward();opt.step()
        self.assertLess((spline(x)-target).square().mean().item(),before*.1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model=tiny();self.ids=torch.tensor([[1,7,9,2,4,6]])
        self.domains=collect_domains(self.model,[self.ids])

    def test_all_qk_replaced_and_model_parity(self):
        with torch.no_grad():before=self.model(self.ids,use_cache=False).logits
        values=[l.self_attn.v_proj for l in self.model.model.layers]
        install(self.model,self.domains)
        self.assertTrue(all(isinstance(m,SplineProjection) for m in projections(self.model).values()))
        self.assertEqual(len(projections(self.model)),4)
        for i,l in enumerate(self.model.model.layers):self.assertIs(l.self_attn.v_proj,values[i])
        with torch.no_grad():after=self.model(self.ids,use_cache=False).logits
        torch.testing.assert_close(before,after,rtol=1e-5,atol=2e-7)
        for name,p in self.model.named_parameters():
            self.assertEqual(p.requires_grad,name.endswith('.coefficients'))

    def test_native_cache_matches_full_sequence(self):
        install(self.model,self.domains)
        with torch.no_grad():
            full=self.model(self.ids,use_cache=False).logits[:,-1]
            first=self.model(self.ids[:,:-1],use_cache=True)
            cached=self.model(self.ids[:,-1:],past_key_values=first.past_key_values,use_cache=True).logits[:,-1]
        torch.testing.assert_close(full,cached,rtol=1e-5,atol=2e-7)

    def test_checkpoint_roundtrip(self):
        install(self.model,self.domains)
        with tempfile.TemporaryDirectory() as d:
            save_splines(Path(d)/'checkpoint',self.model,{'model':'tiny'})
            other=tiny();meta=load_splines(Path(d)/'checkpoint',other)
            self.assertEqual(meta['model'],'tiny')
            with torch.no_grad():
                torch.testing.assert_close(self.model(self.ids,use_cache=False).logits,
                    other(self.ids,use_cache=False).logits,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
