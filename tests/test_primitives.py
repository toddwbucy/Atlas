"""Tests for Atlas pure functions (primitives)."""

import math
import torch
import pytest
from atlas.primitives import polynomial_features, newton_schulz5, omega_loss, lattice_update, lattice_readout


class TestPolynomialFeatures:
    """Tests for polynomial_features (Eq 5)."""

    def test_output_shape_degree2(self):
        """Degree 2 should produce d + d*(d+1)/2 features."""
        d = 16
        x = torch.randn(4, d)
        coeffs = torch.tensor([1.0, 0.5])
        out = polynomial_features(x, degree=2, coeffs=coeffs)
        expected_dim = d + d * (d + 1) // 2  # 16 + 136 = 152
        assert out.shape == (4, expected_dim)

    def test_output_shape_degree1(self):
        """Degree 1 should produce d features (scaled input)."""
        d = 8
        x = torch.randn(3, d)
        coeffs = torch.tensor([1.0])
        out = polynomial_features(x, degree=1, coeffs=coeffs)
        assert out.shape == (3, d)

    def test_cross_terms_present(self):
        """Verify outer product captures cross-terms x_i * x_j."""
        x = torch.tensor([[2.0, 3.0]])
        coeffs = torch.tensor([1.0, 1.0])
        out = polynomial_features(x, degree=2, coeffs=coeffs)
        # Degree 1: [2, 3] * sqrt(1.0)
        # Degree 2: upper triangle of outer product = [4, 6, 9] * sqrt(1.0)
        # Cross-term 2*3=6 must be present
        assert out.shape[-1] == 2 + 3  # d + d*(d+1)/2 = 2 + 3
        # The outer product part should contain the cross-term
        poly_part = out[0, 2:]  # [4, 6, 9]
        assert torch.isclose(poly_part[1], torch.tensor(6.0), atol=1e-5)

    def test_batched_input(self):
        """Should work with arbitrary batch dims."""
        x = torch.randn(2, 8, 4, 16)
        coeffs = torch.tensor([1.0, 0.5])
        out = polynomial_features(x, degree=2, coeffs=coeffs)
        expected = 16 + 16 * 17 // 2
        assert out.shape == (2, 8, 4, expected)

    def test_gradients_flow(self):
        """Coefficients should have gradients."""
        x = torch.randn(4, 8)
        coeffs = torch.tensor([1.0, 0.5], requires_grad=True)
        out = polynomial_features(x, degree=2, coeffs=coeffs)
        out.sum().backward()
        assert coeffs.grad is not None


class TestNewtonSchulz5:
    """Tests for newton_schulz5 (Eq 40)."""

    def test_output_shape(self):
        """Output shape should match input."""
        S = torch.randn(4, 8, 8)
        out = newton_schulz5(S, num_iterations=5)
        assert out.shape == S.shape

    def test_near_orthogonal(self):
        """Result should be approximately orthogonal: X^T X ~ I."""
        S = torch.randn(32, 32) * 0.1
        X = newton_schulz5(S.unsqueeze(0), num_iterations=10).squeeze(0)
        XtX = X.T @ X
        I = torch.eye(32)
        # Should be close to identity (up to scaling)
        # NS5 produces the polar factor which satisfies X^T X = I
        # for well-conditioned inputs
        off_diag = XtX - torch.diag(XtX.diag())
        assert off_diag.abs().max() < 0.5  # Off-diagonal should be small

    def test_non_square_matrix(self):
        """Should work with non-square matrices."""
        S = torch.randn(16, 32)
        out = newton_schulz5(S.unsqueeze(0), num_iterations=5).squeeze(0)
        assert out.shape == (16, 32)

    def test_zero_input(self):
        """Should handle zero input gracefully."""
        S = torch.zeros(4, 8, 8)
        out = newton_schulz5(S, num_iterations=5)
        assert not torch.isnan(out).any()

    def test_batched(self):
        """Should work with batch dimension."""
        S = torch.randn(3, 16, 16)
        out = newton_schulz5(S, num_iterations=5)
        assert out.shape == (3, 16, 16)


class TestOmegaLoss:
    """Tests for omega_loss (Eq 9)."""

    def test_scalar_output(self):
        """Loss should be a scalar."""
        def mem_fn(k):
            return k[:, :4]  # Simple projection

        keys = torch.randn(8, 10)
        values = torch.randn(8, 4)
        gamma = torch.ones(8)
        loss = omega_loss(mem_fn, keys, values, gamma)
        assert loss.dim() == 0  # scalar

    def test_zero_gamma_zero_loss(self):
        """Zero pruning gates should give zero loss."""
        def mem_fn(k):
            return k

        keys = torch.randn(8, 4)
        values = torch.randn(8, 4)
        gamma = torch.zeros(8)
        loss = omega_loss(mem_fn, keys, values, gamma)
        assert loss.item() == 0.0

    def test_perfect_memory_zero_loss(self):
        """If M(k) == v exactly, loss should be zero."""
        values = torch.randn(8, 4)

        def mem_fn(k):
            return values  # Perfect memory

        keys = torch.randn(8, 4)
        gamma = torch.ones(8)
        loss = omega_loss(mem_fn, keys, values, gamma)
        assert loss.item() < 1e-10

    def test_gradients_flow_through(self):
        """Gradients should flow through the loss."""
        W = torch.randn(4, 10, requires_grad=True)

        def mem_fn(k):
            return k @ W.T

        keys = torch.randn(8, 10)
        values = torch.randn(8, 4)
        gamma = torch.ones(8)
        loss = omega_loss(mem_fn, keys, values, gamma)
        loss.backward()
        assert W.grad is not None


class TestLatticeUpdate:
    """Tests for lattice_update (Karami et al. 2504.05646)."""

    def test_output_shape(self):
        """Output shape should match input state."""
        m, d = 16, 64
        S = torch.randn(m, d)
        S = S / S.norm()
        k = torch.randn(m)
        v = torch.randn(d)
        S_new = lattice_update(S, k, v, mu=0.99, eta=0.01)
        assert S_new.shape == (m, d)

    def test_norm_preservation(self):
        """Updated state should be normalized (unit Frobenius norm)."""
        m, d = 16, 64
        S = torch.randn(m, d)
        S = S / S.norm()
        k = torch.randn(m)
        v = torch.randn(d)
        S_new = lattice_update(S, k, v, mu=0.99, eta=0.01)
        assert abs(S_new.norm().item() - 1.0) < 1e-5

    def test_repeated_updates_stable(self):
        """Multiple updates should not cause norm explosion."""
        m, d = 8, 16
        S = torch.randn(m, d)
        S = S / S.norm()
        for _ in range(100):
            k = torch.randn(m)
            v = torch.randn(d)
            S = lattice_update(S, k, v, mu=0.99, eta=0.01)
        assert abs(S.norm().item() - 1.0) < 1e-5
        assert not torch.isnan(S).any()


class TestLatticeReadout:
    """Tests for lattice_readout (Karami et al. 2504.05646)."""

    def test_output_shape(self):
        """Readout should produce [d] vector."""
        m, d = 16, 64
        S = torch.randn(m, d)
        q = torch.randn(m)
        y = lattice_readout(S, q)
        assert y.shape == (d,)
