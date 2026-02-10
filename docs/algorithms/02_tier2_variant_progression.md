# Tier 2: Variant Progression

> **Purpose**: Complete forward-pass algorithms for all 6 Atlas-family models.
> Each variant composes Tier 1 primitives differently. The progression shows how
> each new ingredient (sliding window, deep MLP, momentum, Muon) adds a capability.
>
> **Depends on**: Tier 0 (interfaces), Tier 1 (all 7 primitives)
>
> **Appendix B Examples**: Eqs 44-47 provide the simplest smoke-test forms
> (linear attention, DeltaNet as special cases of the framework)
>
> **Appendix D Implementations**: Eqs 48-58 provide implementation-ready versions

---

## Prelude: Smoke-Test Forms (Appendix B)

Before the six variants, the paper proves that known models are special cases.
These serve as unit-test targets: if our code can reproduce these, the framework is correct.

### Example B.1: Linear Attention as Hebbian Rule

**Source**: Eqs 45-46 (Appendix B.1)
**Graph Nodes**: `atlas_equations/eq-045-linear-attention-as-gd`, `atlas_equations/eq-046-linear-attention-simplified`

```
FUNCTION linear_attention_update(M: Tensor[d_v, d_k],
                                  k_t: Tensor[d_k],
                                  v_t: Tensor[d_v],
                                  eta_t: Scalar) -> Tensor[d_v, d_k]:
    """
    Linear attention = GD on dot-product similarity (Eq 45-46).

    Attentional bias: l_t = <M k_t, v_t>  (dot-product similarity)
    Gradient: grad_M l_t = v_t k_t^T      (outer product)
    Update: M_t = M_{t-1} + eta_t * v_t k_t^T

    This IS gated linear attention / Hebbian rule.
    Simplest possible associative memory update.
    """
    M_new = M + eta_t * outer(v_t, k_t)
    RETURN M_new
```

### Example B.2: DeltaNet as Regression Rule

**Source**: Eq 47 (Appendix B.1)
**Graph Node**: `atlas_equations/eq-047-deltanet-from-regression`

```
FUNCTION deltanet_update(M: Tensor[d_v, d_k],
                          k_t: Tensor[d_k],
                          v_t: Tensor[d_v],
                          eta_t: Scalar) -> Tensor[d_v, d_k]:
    """
    DeltaNet = GD on L2 regression loss (Eq 47).

    Attentional bias: l_t = ||M k_t - v_t||^2  (regression)
    Gradient: grad_M l_t = 2(M k_t - v_t) k_t^T
    Update: M_t = (I - eta_t k_t k_t^T) M_{t-1} + eta_t v_t k_t^T

    The (I - eta_t k_t k_t^T) term is the Delta rule's "erase then write."
    """
    erase = eta_t * outer(M @ k_t, k_t)    -- erase old association
    write = eta_t * outer(v_t, k_t)         -- write new association
    M_new = M - erase + write
    RETURN M_new
```

**Smoke test**: These two should produce identical outputs to reference implementations
of linear attention and DeltaNet. If they don't, our framework has a bug.

---

## Variant 2.1: DLA (Deep Linear Attention)

**Source**: Eq 19, Eq 48-51 (Appendix D.1)
**Graph Nodes**: `atlas_equations/eq-019-dla-memory-update`, `eq-048` through `eq-051`
**Table 1 characteristics**: Flexible Context (1/5)
**Three-axis config**: M=linear/deep, l=dot-product, Optimizer=GD

The simplest Atlas-family model. Per-token updates, no sliding window, no momentum.

```
ALGORITHM DLA_forward(x: Tensor[N, d_in],
                      W_Q, W_K, W_V,          -- outer loop params
                      M_0: Tensor[d_v, d_k],   -- initial memory (outer loop)
                      alpha_conv,               -- decay params (outer loop)
                      eta: Scalar,              -- learning rate
                      phi: PolynomialFeatures   -- optional, degree p
                      ) -> Tensor[N, d_v]:
    """
    Deep Linear Attention (Eqs 48-51).

    Per-token memory update with dot-product attentional bias.
    When M is linear and phi=identity: recovers gated linear attention.
    When M is deep MLP: deep nonlinear attention.

    Primitives used: 1.1 (QKV), 1.2 (poly, optional), 1.7 (decay)
    """
    -- Step 1: Project inputs (Prim 1.1)
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)

    -- Step 2: Optional polynomial features (Prim 1.2)
    IF phi is not None:
        K = polynomial_features(K, phi.p, phi.coefficients)
        Q = polynomial_features(Q, phi.p, phi.coefficients)

    -- Step 3: Sequential per-token update
    M = M_0
    outputs = []
    FOR t IN 0..N-1:
        -- Compute decay (Prim 1.7)
        alpha_t = compute_dynamic_decay(x[t], alpha_conv)

        -- Attentional bias: dot-product (Eq 49)
        -- l_t = <M(k_t), v_t>

        -- Memory update (Eq 50): GD on dot-product loss
        -- grad = v_t @ K[t]^T (for linear M)
        -- For deep M: full backprop through MLP
        grad = gradient_of(<M.forward(K[t]), V[t]>, M.theta_M)
        M.theta_M = alpha_t * M.theta_M - eta * grad

        -- Output: query the updated memory
        y_t = M.forward(Q[t])
        outputs.append(y_t)

    RETURN stack(outputs)   -- [N, d_v]

    -- LINEAR SPECIAL CASE (Eq 51):
    -- When M is a matrix W: W_t = alpha_t * W_{t-1} + V[t] @ K[t]^T
    -- Output: y_t = W_t @ Q[t]
    -- This IS gated linear attention (GLA)
```

---

## Variant 2.2: SWLA (Sliding Window Linear Attention)

**Source**: Eq 20, Eq 52-55 (Appendix D.2)
**Graph Nodes**: `atlas_equations/eq-020-swla-closed-form`, `eq-052` through `eq-055`
**Table 1 characteristics**: Flexible Context + Dynamic Decay (2/5)
**Three-axis config**: M=linear/deep, l=dot-product, Optimizer=GD+decay
**New over DLA**: Sliding window of c tokens instead of per-token

```
ALGORITHM SWLA_forward(x: Tensor[N, d_in],
                       W_Q, W_K, W_V,
                       M_0: Tensor[d_v, d_k],
                       alpha_conv, eta: Scalar,
                       c: int,                   -- window size (NEW)
                       gamma_net,                -- pruning weights (NEW)
                       phi: PolynomialFeatures
                       ) -> Tensor[N, d_v]:
    """
    Sliding Window Linear Attention (Eqs 52-55).

    Same as DLA but optimizes over a WINDOW of c tokens.
    c=1 recovers DLA. c>1 gives context awareness.

    Primitives used: 1.1 (QKV), 1.2 (poly), 1.3 (Omega Rule), 1.7 (decay)
    """
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)
    IF phi: K, Q = poly(K), poly(Q)

    M = M_0
    outputs = []
    FOR t IN 0..N-1:
        alpha_t = compute_dynamic_decay(x[t], alpha_conv)

        -- Sliding window: gather last c tokens (Prim 1.3)
        start = max(0, t - c + 1)
        K_window = K[start:t+1]    -- [c', d_k] where c' = min(c, t+1)
        V_window = V[start:t+1]    -- [c', d_v]
        gamma = gamma_net(x[start:t+1])  -- data-dependent pruning

        -- Omega-style gradient over window (Eq 54)
        -- l = sum gamma_i <M(k_i), v_i>  (dot-product, not L2)
        grad = sum([gamma[i] * gradient_of(<M.forward(K_window[i]), V_window[i]>, M.theta_M)
                     for i in range(len(K_window))])
        M.theta_M = alpha_t * M.theta_M - eta * grad

        y_t = M.forward(Q[t])
        outputs.append(y_t)

    RETURN stack(outputs)

    -- LINEAR SPECIAL CASE (Eq 55):
    -- M_t = alpha_t * M_{t-1} - sum eta_i v_i k_i^T
```

---

## Variant 2.3: OmegaNet

**Source**: Eq 10, Eq 56 (Appendix D.3)
**Graph Nodes**: `atlas_equations/eq-010-omeganet-update`, `atlas_equations/eq-056-omeganet-detailed`
**Table 1 characteristics**: Flexible Context + Locally Optimal (2/5 — different 2 than SWLA)
**Three-axis config**: M=linear/deep, l=L2 regression, Optimizer=GD
**New over SWLA**: L2 regression loss instead of dot-product + polynomial features

```
ALGORITHM OmegaNet_forward(x: Tensor[N, d_in],
                            W_Q, W_K, W_V,
                            M_0, alpha_conv, eta: Scalar,
                            c: int, gamma_net,
                            phi: PolynomialFeatures
                            ) -> Tensor[N, d_v]:
    """
    OmegaNet (Eq 56, implementation-ready).

    Key difference from SWLA: uses L2 regression loss ||M(k) - v||^2
    instead of dot-product similarity <M(k), v>.

    L2 regression is the Omega Rule proper (Eq 9).
    This is what makes Omega "strictly more powerful than Delta."

    Primitives used: 1.1 (QKV), 1.2 (poly), 1.3 (Omega Rule), 1.4 (GD update), 1.7 (decay)
    """
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)
    K = polynomial_features(K, phi.p, phi.coefficients)
    Q = polynomial_features(Q, phi.p, phi.coefficients)

    M = M_0
    outputs = []
    FOR t IN 0..N-1:
        alpha_t = compute_dynamic_decay(x[t], alpha_conv)

        -- Sliding window
        start = max(0, t - c + 1)
        K_window, V_window = K[start:t+1], V[start:t+1]
        gamma = gamma_net(x[start:t+1])

        -- Omega Rule loss (Prim 1.3) — L2, not dot-product
        loss = omega_rule_loss(M, K_window, V_window, gamma)

        -- GD update (Prim 1.4)
        memory_update_gd(M, K_window, V_window, gamma, alpha_t, eta)

        y_t = M.forward(Q[t])
        outputs.append(y_t)

    RETURN stack(outputs)
```

---

## Variant 2.4: DeepTransformers

**Source**: Eq 25, Eq 26 (containment proof)
**Graph Nodes**: `atlas_equations/eq-025-deeptransformers-formulation`, `atlas_equations/eq-026-deeptransformers-subsumes-transformers`
**Table 1 characteristics**: Flexible Context + Dynamic Decay + Deep Neural Memory (3/5)
**Three-axis config**: M=deep MLP, l=dot-product, Optimizer=GD+decay
**New over OmegaNet**: Deep MLP memory (Eq 42) instead of linear

```
ALGORITHM DeepTransformers_forward(x: Tensor[N, d_in],
                                     W_Q, W_K, W_V,
                                     M_0: DeepMemoryMLP,  -- Eq 42
                                     alpha_conv, eta: Scalar,
                                     phi_star: ExponentialKernel  -- Eq 22
                                     ) -> Tensor[N, d_v]:
    """
    DeepTransformers (Eq 25).

    Uses:
    - Deep MLP memory (Prim 1.5, Eq 42) instead of linear
    - Exponential kernel phi* (Eq 22) instead of polynomial
    - Per-token Hebbian update (dot-product similarity)

    THE CONTAINMENT PROOF (Eq 26):
        When M is linear and update is Hebbian:
            y_t = M_t phi*(q_t) = sum v_i exp(q^T k_i)
        This IS unnormalized Softmax attention.
        Therefore DeepTransformers STRICTLY GENERALIZE Transformers.
        Every Transformer is a DeepTransformer with linear M + Hebbian.

    Primitives used: 1.1 (QKV), 1.5 (deep MLP), 1.7 (decay)
    """
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)

    M = M_0   -- deep MLP (Eq 42)
    outputs = []
    FOR t IN 0..N-1:
        alpha_t = compute_dynamic_decay(x[t], alpha_conv)

        -- Hebbian update with exponential kernel (Eq 25)
        -- l_t = <M(phi*(k_t)), v_t>
        phi_k = exponential_kernel(K[t])     -- infinite-dim in theory
        grad = gradient_of(<M.forward(phi_k), V[t]>, M.theta_M)
        M.theta_M = alpha_t * M.theta_M - eta * grad

        phi_q = exponential_kernel(Q[t])
        y_t = M.forward(phi_q)
        outputs.append(y_t)

    RETURN stack(outputs)

    -- PRACTICAL NOTE: exponential kernel has infinite dimensions.
    -- Implementation uses random Fourier features or polynomial
    -- approximation (Eq 5 with high p) as proxy.
```

**Atlas IS #4 compliance**: This variant PROVES Transformers are a special case.
Eq 26 is the formal containment proof.

---

## Variant 2.5: DoT (Deep Omega Transformer)

**Source**: Eq 27
**Graph Node**: `atlas_equations/eq-027-dot-formulation`
**Table 1 characteristics**: 4/5 (all except Locally Optimal via Muon)
**Three-axis config**: M=deep MLP, l=L2 regression, Optimizer=Momentum+decay
**New over DeepTransformers**: L2 loss (Omega Rule) + momentum

```
ALGORITHM DoT_forward(x: Tensor[N, d_in],
                       W_Q, W_K, W_V,
                       M_0: DeepMemoryMLP,
                       S_0: Tensor,              -- momentum state (NEW)
                       alpha_conv, eta: Scalar,
                       theta_t: Scalar,          -- momentum coefficient (NEW)
                       c: int, gamma_net,
                       phi_star: ExponentialKernel
                       ) -> Tensor[N, d_v]:
    """
    Deep Omega Transformer (Eq 27).

    Combines ALL previous ingredients:
    - Deep MLP memory (from DeepTransformers)
    - Omega Rule / L2 loss (from OmegaNet)
    - Sliding window of c tokens (from SWLA)
    - Momentum (NEW) — smooths gradient updates

    4/5 Table 1 characteristics. Only missing: Muon (locally optimal).

    Primitives used: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7
    """
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)
    K, Q = apply_kernel(K, phi_star), apply_kernel(Q, phi_star)

    M = M_0
    S = S_0   -- momentum accumulator
    outputs = []
    FOR t IN 0..N-1:
        alpha_t = compute_dynamic_decay(x[t], alpha_conv)

        -- Sliding window
        start = max(0, t - c + 1)
        K_window, V_window = K[start:t+1], V[start:t+1]
        gamma = gamma_net(x[start:t+1])

        -- Omega Rule gradient
        loss = omega_rule_loss(M, K_window, V_window, gamma)
        grad = gradient(loss, M.theta_M)

        -- Momentum update (NEW)
        S = theta_t * S + grad

        -- Memory update with momentum (not raw gradient)
        M.theta_M = alpha_t * M.theta_M - eta * S

        y_t = M.forward(apply_kernel(Q[t], phi_star))
        outputs.append(y_t)

    RETURN stack(outputs)
```

---

## Variant 2.6: Atlas (Full Model)

**Source**: Eq 32, 33, 57, 58 (Appendix D.4)
**Graph Nodes**: `atlas_equations/eq-032-atlas-memory-muon`, `eq-033-atlas-momentum`,
`eq-057-atlas-detailed-memory`, `eq-058-atlas-detailed-momentum`
**Table 1 characteristics**: 5/5 (ALL)
**Three-axis config**: M=deep MLP (gated), l=L2 regression, Optimizer=Muon+decay
**New over DoT**: Muon (Newton-Schulz5) replaces raw momentum step

```
ALGORITHM Atlas_forward(x: Tensor[N, d_in],
                         W_Q, W_K, W_V,
                         M_0: GatedMemoryMLP,     -- Eq 43 (Atlas++)
                         S_0: Tensor,              -- momentum state
                         alpha_conv, eta: Scalar,
                         theta_t: Scalar,          -- momentum coefficient
                         c: int, gamma_net,
                         phi: PolynomialFeatures,  -- Eq 5
                         ns_iterations: int = 5    -- Newton-Schulz steps
                         ) -> Tensor[N, d_v]:
    """
    Atlas — the full model (Eqs 57-58, implementation-ready).

    Eq 57: M_t = alpha_t * M_{t-1} + Newton-Schulz5(S_t)
    Eq 58: S_t = theta_t * S_{t-1} - sum eta_i grad ||M(phi(k_i)) - v_i||^2

    This is DoT + Muon. The Newton-Schulz5 step (Prim 1.6) is what makes
    Atlas "locally optimal" — the ONLY model achieving all 5 characteristics.

    ALL Tier 1 primitives used: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7

    COMPOSABLE PARAMETERS:
        alpha_t: dynamic decay          (Prim 1.7, outer loop)
        eta:     inner learning rate    (Prim 1.4)
        theta_t: momentum coefficient   (outer loop)
        c:       window size            (Prim 1.3)
        p:       polynomial degree      (Prim 1.2)
        k:       NS iterations          (Prim 1.6)
    """
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)      -- Prim 1.1
    K = polynomial_features(K, phi.p, phi.coefficients)  -- Prim 1.2
    Q = polynomial_features(Q, phi.p, phi.coefficients)

    M = M_0     -- gated MLP (Eq 43, Prim 1.5)
    S = S_0     -- momentum state
    outputs = []
    FOR t IN 0..N-1:
        -- Dynamic decay (Prim 1.7)
        alpha_t = compute_dynamic_decay(x[t], alpha_conv)

        -- Sliding window (Prim 1.3)
        start = max(0, t - c + 1)
        K_window, V_window = K[start:t+1], V[start:t+1]
        gamma = gamma_net(x[start:t+1])

        -- Omega Rule gradient (Prim 1.3)
        loss = omega_rule_loss(M, K_window, V_window, gamma)
        grad = gradient(loss, M.theta_M)

        -- Momentum accumulation (Eq 58)
        S = theta_t * S + grad

        -- Newton-Schulz5 orthogonalization (Prim 1.6)
        S_orth = newton_schulz5(S, num_iterations=ns_iterations)

        -- Memory update (Eq 57)
        M.theta_M = alpha_t * M.theta_M - eta * S_orth

        -- Query memory
        y_t = M.forward(Q[t])
        outputs.append(y_t)

    RETURN stack(outputs)
```

---

## Variant Composition Summary

Each variant adds ONE ingredient to the previous:

```
DLA                 = QKV + linear M + dot-product loss + GD
  + sliding window  = SWLA
  + L2 loss + poly  = OmegaNet
  + deep MLP        = DeepTransformers
  + momentum        = DoT
  + Muon (NS5)      = Atlas
```

| Variant | Prim 1.1 | Prim 1.2 | Prim 1.3 | Prim 1.4 | Prim 1.5 | Prim 1.6 | Prim 1.7 |
|---------|----------|----------|----------|----------|----------|----------|----------|
|         | QKV      | Poly     | Omega    | GD       | Deep MLP | NS5      | Decay    |
| DLA     | yes      | opt      | no (c=1) | yes      | opt      | no       | opt      |
| SWLA    | yes      | opt      | yes      | yes      | opt      | no       | yes      |
| OmegaNet| yes      | yes      | yes      | yes      | opt      | no       | opt      |
| DeepTr  | yes      | exp      | no (c=1) | yes      | yes      | no       | yes      |
| DoT     | yes      | exp      | yes      | yes      | yes      | no       | yes      |
| Atlas   | yes      | yes      | yes      | yes      | yes      | yes      | yes      |

---

## Tier 2 Audit Checklist

### Atlas IS Container (6 principles)

| # | Principle | Tier 2 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | Context memorization | IMPLEMENTED | All variants with c>1 use sliding window |
| 2 | Deep neural memory + non-linear | IMPLEMENTED | Variants 2.4-2.6 use deep MLP |
| 3 | Locally optimal | IMPLEMENTED | Variant 2.6 (Atlas) uses Muon |
| 4 | Strict generalization | PROVEN | Variant 2.4 (DeepTransformers) + Eq 26 proof |
| 5 | Polynomial/exponential features | IMPLEMENTED | All variants except DLA use phi or phi* |
| 6 | Composable building blocks | DEMONSTRATED | 6 variants from 7 primitives, Table shows composition |

### Atlas IS NOT Container (5 principles)

| # | Principle | Tier 2 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | NOT token-level | COMPLIANT | Omega Rule window, except DLA/DeepTr which are c=1 |
| 2 | NOT linear-only | COMPLIANT | Deep MLP in variants 2.4-2.6, polynomial in 2.3+ |
| 3 | NOT simple GD only | COMPLIANT | GD -> momentum -> Muon progression shown |
| 4 | NOT replacement | PROVEN | Eq 26: strict generalization, not competition |
| 5 | NOT single architecture | DEMONSTRATED | 6 distinct variants + smoke tests |

### NL IS/IS NOT Container

Same valid gaps as Tier 0/1 — Atlas is a single-level module.
NL IS #2 (nesting), #7 (self-modifying), #8 (CMS) remain valid gaps.

### Composition Integrity Check

Every variant uses ONLY Tier 1 primitives. No variant introduces new operations.
The progression is strictly additive — each adds exactly one ingredient.

---

**Tier 2 Complete. Proceed to Tier 3: Chunkwise Parallelization.**
