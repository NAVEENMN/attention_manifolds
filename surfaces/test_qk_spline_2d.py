import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from qk_spline_2d import QKSpline2D, adapters, calibrate, cubic_basis, install, load_adapter, save_adapter


def tiny_model():
    torch.manual_seed(10)
    config = LlamaConfig(vocab_size=37, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0.0, use_cache=False)
    config._attn_implementation = 'eager'
    return LlamaForCausalLM(config).float().eval()


class SplineTests(unittest.TestCase):
    def test_basis_partition_endpoints_and_gradients(self):
        for m in (4, 8, 12):
            x = torch.linspace(0, 1, 101, dtype=torch.float64, requires_grad=True)
            b = cubic_basis(x, m)
            torch.testing.assert_close(b.sum(-1), torch.ones_like(x), atol=1e-12, rtol=1e-12)
            self.assertGreaterEqual(b.min().item(), -1e-12)
            self.assertTrue(torch.isfinite(torch.autograd.grad(b.square().sum(), x)[0]).all())
        x = torch.tensor([0.13, 0.41, 0.83], dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda t: cubic_basis(t, 8), (x,)))

    def test_contraction_against_explicit_tensor_product(self):
        a = QKSpline2D(3, 5)
        with torch.no_grad():
            a.coefficients.normal_()
        q, k = torch.randn(2, 2, 4, 3), torch.randn(2, 2, 5, 3)
        actual = a(q, k)
        bq, bk = a.basis(q, 'q'), a.basis(k, 'k')
        expected = torch.zeros_like(actual)
        for d in range(3):
            for i in range(5):
                for j in range(5):
                    expected += a.coefficients[d, i, j] * bq[..., d, i, None] * bk[..., d, j].unsqueeze(-2)
        torch.testing.assert_close(actual, expected / 3**0.5, atol=3e-7, rtol=1e-5)

    def test_local_edit_support_head_scope_and_restore(self):
        a = QKSpline2D(1, 8, radius=1)
        q = torch.tensor([-.8, 0., .8]).view(1, 1, 3, 1).expand(1, 2, 3, 1)
        base = a(q, q).detach()
        with self.assertRaises(RuntimeError):
            with a.local_edit(1, 0, 3, 3, 2):
                diff = a(q, q) - base
                torch.testing.assert_close(diff[:, 0], torch.zeros_like(diff[:, 0]))
                self.assertGreater(diff[0, 1, 1, 1].item(), 0)
                self.assertEqual(diff[0, 1, 2].abs().sum().item(), 0)
                self.assertEqual(diff[0, 1, :, 2].abs().sum().item(), 0)
                raise RuntimeError('restore test')
        torch.testing.assert_close(a(q, q), base, rtol=0, atol=0)

    def test_row_constant_softmax_gauge(self):
        logits = torch.randn(2, 3, 4, 4)
        shift = torch.randn(2, 3, 4, 1)
        torch.testing.assert_close(logits.softmax(-1), (logits + shift).softmax(-1))

    def test_uniform_coefficient_offset_is_softmax_null(self):
        adapter = QKSpline2D(3, 8)
        q, k = torch.randn(1, 2, 5, 3), torch.randn(1, 2, 5, 3)
        logits = torch.randn(1, 2, 5, 5)
        with torch.no_grad():
            adapter.coefficients.fill_(2)
        correction = adapter(q, k)
        torch.testing.assert_close(correction, torch.full_like(correction, 2 * 3**0.5))
        torch.testing.assert_close(logits.softmax(-1), (logits + correction).softmax(-1))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.ids = torch.tensor([[1, 7, 9, 2, 4, 6], [1, 5, 3, 8, 2, 4]])

    def test_zero_identity_calibration_and_checkpoint(self):
        with torch.no_grad():
            original = self.model(self.ids).logits
        install(self.model)
        stats = calibrate(self.model, [self.ids])
        self.assertEqual(stats['0.q']['samples'], 48)
        with torch.no_grad():
            torch.testing.assert_close(self.model(self.ids).logits, original, atol=1e-7, rtol=1e-6)
        with torch.no_grad():
            adapters(self.model)[0].coefficients.normal_(std=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adapter.pt'
            save_adapter(path, self.model, {'model': 'tiny-test'})
            fresh = tiny_model()
            self.assertEqual(load_adapter(path, fresh)['model'], 'tiny-test')
            torch.testing.assert_close(self.model(self.ids).logits, fresh(self.ids).logits, rtol=0, atol=0)

    def test_only_surfaces_train_and_initial_gradient_is_nonzero(self):
        install(self.model)
        loss = self.model(self.ids, labels=self.ids).loss
        loss.backward()
        for name, p in self.model.named_parameters():
            if name.endswith('qk_spline.coefficients'):
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)
            else:
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)

    def test_causality_after_training_and_edit(self):
        install(self.model)
        with torch.no_grad():
            for a in adapters(self.model).values():
                a.coefficients.normal_()
        modified = self.ids.clone()
        modified[:, 4:] = torch.tensor([11, 12])
        with adapters(self.model)[0].local_edit(2, 1, 3, 4, 10), torch.no_grad():
            x, y = self.model(self.ids).logits, self.model(modified).logits
        torch.testing.assert_close(x[:, :4], y[:, :4], atol=0, rtol=0)

    def test_padded_batch_matches_unpadded(self):
        install(self.model)
        with torch.no_grad():
            adapters(self.model)[0].coefficients.normal_()
            padded = self.model(self.ids[:1], attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]])).logits
            short = self.model(self.ids[:1, :3]).logits
        torch.testing.assert_close(padded[:, :3], short, atol=1e-7, rtol=1e-6)

    def test_cache_rejected(self):
        install(self.model)
        with self.assertRaisesRegex(ValueError, 'cache'):
            self.model(self.ids, use_cache=True)

    def test_small_training_smoke(self):
        install(self.model)
        params = [a.coefficients for a in adapters(self.model).values()]
        optimizer = torch.optim.Adam(params, lr=.1)
        before = self.model(self.ids, labels=self.ids).loss.item()
        for _ in range(8):
            optimizer.zero_grad()
            loss = self.model(self.ids, labels=self.ids).loss
            loss.backward()
            optimizer.step()
        after = self.model(self.ids, labels=self.ids).loss.item()
        self.assertLess(after, before)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
