# Tier 1: Shared Primitives

> **Purpose**: Concrete, reusable operations that appear across multiple Atlas variants.
> These ARE executable — each maps directly to a function.
>
> **Depends on**: Tier 0 (AssociativeMemory interface, Inner/Outer partition)
>
> **Graph Sources**: atlas_equations (Eqs 4, 5, 9, 10, 22, 40, 42, 43, 48)
> atlas_definitions (def-polynomial-feature-mapping, def-omega-rule, def-deep-mlp-memory,
> def-memory-optimizer-spectrum, def-memory-capacity)
> atlas_abstractions (impl-newton-schulz5)
> **Cross-paper**: hope_equations/eq-042 (Muon), eq-044 (Newton-Schulz iteration)

---

## Primitive 1.1: QKV Projections

**Source**: Eq 48 (Appendix D.1), Eq 52 (Appendix D.2)
**Graph Nodes**: `atlas_equations/eq-048-dla-qkv-projections`, `atlas_equations/eq-052-swla-qkv-projections`

Standard linear projections from input to query/key/value spaces.
Identical across all Atlas variants — the projections are OUTER LOOP parameters.

```
FUNCTION qkv_project(x: Tensor[N, d_in],
                     W_Q: Tensor[d_in, d_k],
                     W_K: Tensor[d_in, d_k],
                     W_V: Tensor[d_in, d_v]) -> (Tensor[N, d_k], Tensor[N, d_k], Tensor[N, d_v]):
    """
    Project input to query, key, value spaces.
    Q, K, V = x W_Q, x W_K, x W_V

    OUTER LOOP: W_Q, W_K, W_V are learned via backpropagation.
    These do NOT change during test-time memory updates.
    """
    Q = x @ W_Q    -- queries: what to retrieve
    K = x @ W_K    -- keys: what to index by
    V = x @ W_V    -- values: what to store
    RETURN (Q, K, V)
```

**Invariant**: QKV projections are identical across DLA, SWLA, OmegaNet, DeepTransformers,
DoT, and Atlas. The graph confirms Eq 48 == Eq 52 in description.

---

## Primitive 1.2: Polynomial Feature Mapping

**Source**: Eq 4 (Taylor approximation), Eq 5 (polynomial kernel)
**Graph Nodes**: `atlas_equations/eq-005-polynomial-kernel`, `atlas_definitions/def-polynomial-feature-mapping`
**Definition**: `atlas_definitions/def-polynomial-feature-mapping`

Lifts input from d_k dimensions to a higher-dimensional polynomial space,
amplifying memory capacity from O(d_k) to O(d_k^p).

```
FUNCTION polynomial_features(x: Tensor[..., d_k],
                             p: int,
                             coefficients: Tensor[p+1]) -> Tensor[..., d_poly]:
    """
    Generalized polynomial kernel: phi_p(x)

    Given input x, compute polynomial features up to degree p:
        phi_p(x) = [a_0, a_1 * x, a_2 * (x (x) x), ..., a_p * x^{(x)p}]

    where (x) denotes the Kronecker (outer) product and a_i are learnable
    coefficients initialized to 1/i! (matching the Taylor series of exp).

    Key insight from Eq 5: a_i are LEARNABLE. If a_i -> 0, degree-i features
    are excluded. This is INPUT GATING on the feature space, not on memory.

    Capacity: O(d_k^p) — super-linear when p >= 2
    """
    REQUIRE p >= 1
    REQUIRE coefficients.shape == [p + 1]

    -- Degree 0: constant term
    features = [coefficients[0] * ones_like(x[..., :1])]

    -- Degree 1 through p: successive Kronecker products
    x_power = x                              -- x^{(x)1}
    FOR degree IN 1..p:
        features.append(coefficients[degree] * x_power)
        IF degree < p:
            x_power = kronecker(x_power, x)  -- x^{(x)(degree+1)}

    RETURN concatenate(features, dim=-1)

INITIALIZATION:
    -- Taylor series initialization (matches exp kernel approximation)
    FOR i IN 0..p:
        coefficients[i] = 1.0 / factorial(i)
```

**Connection to Eq 4**: The Taylor series of exp(q^T k) = sum_{i=0}^{inf} (q^T k)^i / i!
When we truncate at degree p and make coefficients learnable, we get Eq 5.
The infinite limit (p -> inf) recovers the exponential kernel of Softmax attention.

**Connection to Eq 22**: The exponential kernel phi*(x) = [1, x/sqrt(1!), x^{ox2}/sqrt(2!), ...]
is the target that polynomial features approximate with finite p.

**Practical note**: For implementation, the full Kronecker product is O(d_k^p) in memory.
Real implementations use random feature approximations or structured products.
The pseudocode shows the mathematical definition; implementation may use approximations.

---

## Primitive 1.3: Omega Rule (Sliding Window Objective)

**Source**: Eq 8 (sliding window objective), Eq 9 (Omega Rule)
**Graph Nodes**: `atlas_equations/eq-008-sliding-window-objective`, `atlas_equations/eq-009-omega-rule`
**Definition**: `atlas_definitions/def-omega-rule`

The core learning rule of Atlas. Named Omega — strictly more powerful than Delta.
Measures surprise of a LOCAL CONTEXT WINDOW, not a single token.

```
FUNCTION omega_rule_loss(M: AssociativeMemory,
                         K_window: Tensor[c, d_k],
                         V_window: Tensor[c, d_v],
                         gamma: Tensor[c]) -> Scalar:
    """
    The Omega Rule (Eq 9, boxed in paper):

        L = sum_{i=t-c+1}^{t} gamma_i^(t) * ||M(k_i) - v_i||^2_2

    Where:
        c >= 1 is the context window size
        gamma_i in [0, 1] are data-dependent pruning weights
        M(k_i) is the memory's prediction for key k_i
        v_i is the target value

    This generalizes:
        c = 1:   recovers Delta rule (Titans)
        c = inf: recovers global least-squares (Eq 7)
        1 < c < inf: the Omega sweet spot — local context awareness

    The gradient of this sum IS the 'surprise' of the context.
    High loss = unfamiliar context = large gradient = more memory update.
    """
    REQUIRE c >= 1
    REQUIRE gamma.shape == [c]
    REQUIRE all(0 <= gamma_i <= 1 for gamma_i in gamma)

    predictions = M.forward(K_window)        -- [c, d_v]
    residuals = predictions - V_window       -- [c, d_v]
    per_token_loss = sum(residuals ** 2, dim=-1)  -- [c]
    weighted_loss = sum(gamma * per_token_loss)    -- scalar

    RETURN weighted_loss
```

**Connection to Eq 6**: Per-token (online) objective `l(M; k_t, v_t) + Ret(M, M_{t-1})`
is the c=1 special case. This is what Titans does.

**Connection to Eq 8**: Eq 8 is the general sliding window form before specifying
the attentional bias l(.). Eq 9 instantiates l(.) = ||M(k) - v||^2_2 (l2 regression).

---

## Primitive 1.4: Memory Update Step (GD-based)

**Source**: Eq 10 (OmegaNet update)
**Graph Node**: `atlas_equations/eq-010-omeganet-update`

The basic memory update: compute Omega Rule gradient, apply with dynamic decay.
This is the GD-based version; Muon (Primitive 1.6) replaces the gradient step.

```
FUNCTION memory_update_gd(M: AssociativeMemory,
                          K_window: Tensor[c, d_k],
                          V_window: Tensor[c, d_v],
                          gamma: Tensor[c],
                          alpha_t: Scalar,
                          lr: Scalar) -> None:
    """
    OmegaNet-style memory update (Eq 10):

        M_t = alpha_t * M_{t-1} - lr * grad_M [sum gamma_i ||M(phi(k_i)) - v_i||^2]

    Where:
        alpha_t = dynamic decay (retention) — sigmoid(conv(x_t)), per-token
        lr = learning rate (inner loop)
        The gradient is w.r.t. M's parameters theta_M

    Two operations fused:
        1. FORGET: alpha_t * M_{t-1}  (decay old memory)
        2. LEARN:  - lr * grad         (absorb new context)

    This is INNER LOOP: only theta_M changes. All other params frozen.
    """
    -- Compute Omega Rule loss
    loss = omega_rule_loss(M, K_window, V_window, gamma)

    -- Compute gradient w.r.t. memory parameters
    grad = gradient(loss, M.theta_M)

    -- Apply update: forget + learn
    M.theta_M = alpha_t * M.theta_M - lr * grad
```

---

## Primitive 1.5: Deep Memory MLP

**Source**: Eq 42 (deep memory MLP), Eq 43 (gated MLP / Atlas++)
**Graph Nodes**: `atlas_equations/eq-042-deep-memory-mlp`, `atlas_equations/eq-043-gated-mlp-atlas-plus`
**Definition**: `atlas_definitions/def-deep-mlp-memory`

Concrete memory architectures that implement the AssociativeMemory interface.

```
FUNCTION deep_memory_mlp(x: Tensor[..., d_k],
                         W_1: Tensor[d_hidden, d_v],
                         W_2: Tensor[d_k, d_hidden]) -> Tensor[..., d_v]:
    """
    Deep memory (Eq 42): M(x) = x + W_1 * sigma(W_2 * x)

    2-layer MLP with residual connection.
    - W_2 projects input to hidden space
    - sigma is nonlinear activation (GELU recommended)
    - W_1 projects back to output space
    - Residual connection (+x) ensures gradient flow

    Capacity: O(d_k * d_o * sum(min(d_h terms))) per Theorem 1
    This is STRICTLY better than linear memory's O(d_k).

    INNER LOOP: W_1, W_2 are theta_M — updated per-chunk via Omega Rule.
    """
    hidden = sigma(W_2 @ x)    -- [d_hidden]
    output = W_1 @ hidden       -- [d_v]
    RETURN x + output           -- residual connection


FUNCTION gated_memory_mlp(x: Tensor[..., d_k],
                          W_1: Tensor[d_hidden, d_v],
                          W_2: Tensor[d_k, d_hidden],
                          W_3: Tensor[d_k, d_hidden]) -> Tensor[..., d_v]:
    """
    Gated MLP memory / Atlas++ (Eq 43): M(x) = x + W_1 * (sigma(W_2 * x) (*) W_3 * x)

    Extends Eq 42 with element-wise gating (SwiGLU-style):
    - sigma(W_2 * x) provides activation
    - W_3 * x provides gating signal
    - (*) is element-wise (Hadamard) product
    - W_1 projects gated-activated hidden to output

    This is the RECOMMENDED architecture — "Atlas++" in experiments.
    More expressive than simple MLP due to gating.

    INNER LOOP: W_1, W_2, W_3 are ALL theta_M — updated per-chunk.
    """
    activated = sigma(W_2 @ x)    -- [d_hidden]
    gate = W_3 @ x                -- [d_hidden]
    gated = activated * gate      -- element-wise product [d_hidden]
    output = W_1 @ gated          -- [d_v]
    RETURN x + output             -- residual connection
```

**Key distinction**: Both are INNER LOOP parameters. During test-time memory updates,
the Omega Rule gradient flows through these MLPs and updates W_1, W_2 (and W_3 for gated).
This is fundamentally different from a standard FFN layer where weights are fixed after training.

---

## Primitive 1.6: Newton-Schulz5 Orthogonalization (Muon)

**Source**: Eq 40 (Newton-Schulz5 parallel)
**Graph Nodes**: `atlas_equations/eq-040-newton-schulz5-parallel`, `atlas_abstractions/impl-newton-schulz5`
**Cross-paper**: `hope_equations/eq-042-muon-optimizer`, `hope_equations/eq-044-newton-schulz-iteration`

The second-order optimizer step that makes Atlas "locally optimal."
Replaces raw gradient/momentum with its orthogonalized polar factor.

```
FUNCTION newton_schulz5(S: Tensor[d_out, d_in],
                        num_iterations: int = 5) -> Tensor[d_out, d_in]:
    """
    Newton-Schulz5: orthogonalize momentum matrix S.

    Given momentum S (accumulated gradients), compute S' approx U V^T
    where S = U Sigma V^T is the SVD of S.

    This approximates the polar decomposition WITHOUT computing SVD.

    Algorithm:
        X_0 = S / ||S||_F                    (normalize)
        FOR i = 0..num_iterations-1:
            X_{i+1} = X_i (a*I + b*X_i^T X_i + c*(X_i^T X_i)^2)
        S' = X_{num_iterations}

    Coefficients (from Muon paper, Jordan et al. 2024):
        a = 3.4445
        b = -4.7750
        c = 2.0315

    Properties:
        1. k=5 iterations empirically sufficient
        2. Parallelizable: all S_t computed independently (Eq 39)
        3. Prevents memory collapse (well-conditioned updates)
        4. Table 6 ablation: removing Muon degrades ppl 19.04 -> 19.65

    WHY this matters for memory:
        Raw gradient points toward loss reduction but may be ill-conditioned.
        NS5 maps it to the nearest orthogonal matrix — the BEST direction.
        This is what makes Atlas "locally optimal" (Table 1, characteristic #4).
    """
    REQUIRE num_iterations >= 1
    CONSTANTS a = 3.4445, b = -4.7750, c = 2.0315

    -- Normalize
    X = S / frobenius_norm(S)

    -- Iterate
    FOR i IN 0..num_iterations-1:
        XtX = X^T @ X                        -- [d_in, d_in]
        X = X @ (a * I + b * XtX + c * XtX @ XtX)   -- [d_out, d_in]

    RETURN X
```

**Cross-paper identity**: This is the SAME algorithm as HOPE Eq 42 (Muon) and
HOPE Eq 44 (Newton-Schulz iteration step). The coefficients are identical.
Atlas applies it to memory momentum; HOPE applies it to M3 optimizer momentum.

---

## Primitive 1.7: Dynamic Decay (Retention Gate)

**Source**: Implicit in Eq 10 (alpha_t term), explicit in Table 1
**Definition**: Part of `atlas_definitions/def-omega-rule` (gamma weights) and
three-axis model (Axis 3, dynamic decay)

Per-token retention gating: controls how much old memory is kept vs. replaced.

```
FUNCTION compute_dynamic_decay(x: Tensor[N, d_in],
                               conv_weight: Tensor[d_in, kernel_size],
                               bias: Tensor[1]) -> Tensor[N, 1]:
    """
    Dynamic decay alpha_t = sigmoid(short_conv(x_t))

    Computes per-token retention factor:
        alpha_t close to 1: retain previous memory (familiar context)
        alpha_t close to 0: reset memory (novel context)

    The short convolution provides local temporal context —
    alpha_t depends on nearby tokens, not just the current one.

    OUTER LOOP: conv_weight and bias are learned via backpropagation.
    They do NOT change at test time.
    """
    -- Short causal convolution (typically kernel_size=3 or 4)
    conv_out = causal_conv1d(x, conv_weight)   -- [N, 1]
    alpha = sigmoid(conv_out + bias)            -- [N, 1], range (0, 1)

    RETURN alpha

    -- Models WITHOUT dynamic decay: DLA, OmegaNet (alpha_t = 1 always)
    -- Models WITH dynamic decay: SWLA, DeepTransformers, DoT, Atlas
```

**Key insight**: Dynamic decay is what MIRAS calls the "retention gate."
Atlas folds it into Axis 3 (optimizer) rather than giving it a separate axis.
From Table 1, this is the FIRST characteristic that separates DLA from SWLA.

---

## Tier 1 Audit Checklist

### Atlas IS Container (6 principles)

| # | Principle | Tier 1 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | Context memorization | IMPLEMENTED | Prim 1.3: Omega Rule uses window of c tokens |
| 2 | Deep neural memory + non-linear capacity | IMPLEMENTED | Prim 1.2 (polynomial, O(d_k^p)), Prim 1.5 (deep MLP, O(d_k*d_o*sum)) |
| 3 | Locally optimal via advanced optimizers | IMPLEMENTED | Prim 1.6 (Newton-Schulz5 / Muon) |
| 4 | Strict generalization of Transformers | DEFERRED | Proven in Tier 2 via DeepTransformers variant |
| 5 | Polynomial/exponential features | IMPLEMENTED | Prim 1.2 with learnable coefficients a_i |
| 6 | Composable building blocks | VERIFIED | All 7 primitives compose independently |

### Atlas IS NOT Container (5 principles)

| # | Principle | Tier 1 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | NOT token-level | COMPLIANT | Prim 1.3: c >= 1 window, c=1 degenerate case only |
| 2 | NOT linear-only capacity | COMPLIANT | Prim 1.2: O(d_k^p); Prim 1.5: deep MLP |
| 3 | NOT simple GD only | COMPLIANT | Prim 1.4 (GD), Prim 1.6 (Muon) — spectrum provided |
| 4 | NOT replacement | DEFERRED | Tier 2 |
| 5 | NOT single architecture | COMPLIANT | Primitives compose into 6+ variants |

### NL IS Container (8 principles)

| # | Principle | Tier 1 Status | Notes |
|---|-----------|---------------|-------|
| 1 | New learning paradigm | COMPLIANT | Memory-as-optimization realized by Prims 1.3+1.4 |
| 2 | Nested multi-level | VALID GAP | Atlas is one level; nesting is HOPE's job |
| 3 | Own context flow | COMPLIANT | Omega Rule window IS the context flow |
| 4 | Compressing context | COMPLIANT | M* minimizes reconstruction loss = compression |
| 5 | ICL emerges | COMPLIANT | Inner loop (Prim 1.4) IS in-context learning |
| 6 | Optimizers are memory | COMPLIANT | Prim 1.6: Muon IS memory optimizer |
| 7 | Self-modifying | VALID GAP | HOPE-specific (GGD) |
| 8 | Continuum memory | VALID GAP | HOPE-specific (CMS) |

### NL IS NOT Container (5 anti-principles)

| # | Anti-Principle | Tier 1 Status | Notes |
|---|---------------|---------------|-------|
| 1 | NOT single-level | VALID GAP | Atlas is one level by design |
| 2 | NOT shared context | COMPLIANT | Inner/outer have separate flows |
| 3 | NOT static rules | COMPLIANT | Dynamic decay (Prim 1.7), learnable coefficients (Prim 1.2) |
| 4 | NOT discrete memory | COMPLIANT | Continuous parameter space |
| 5 | NOT optimizers as just optimizers | COMPLIANT | Muon IS the memory update mechanism |

### Cross-Paper Verification

| Primitive | Atlas Equation | Cross-Paper Identity | Status |
|-----------|---------------|---------------------|--------|
| 1.6 (NS5) | eq-040 | hope_equations/eq-042 (Muon), eq-044 (NS step) | CONFIRMED |
| 1.3 (Omega) | eq-009 | titans: c=1 special case (atlas_equations/eq-012) | CONFIRMED |
| 1.2 (Poly) | eq-005 | eq-022 (exponential kernel = limit p->inf) | CONFIRMED |
| 1.1 (QKV) | eq-048/052 | Standard across all variants | CONFIRMED |

---

**Tier 1 Complete. Proceed to Tier 2: Variant Progression (DLA -> SWLA -> OmegaNet -> DeepTransformers -> DoT -> Atlas).**
