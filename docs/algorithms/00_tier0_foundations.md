# Tier 0: Foundations

> **Purpose**: Abstract interfaces and structural definitions that everything else builds on.
> These are NOT executable — they define the vocabulary and contracts for Tiers 1-4.
>
> **Graph Sources**: atlas_definitions (def-associative-memory, def-inner-outer-loop-partition,
> def-three-axis-model-description, def-five-memory-characteristics)
> **Equation Sources**: atlas_equations/eq-003-associative-memory-objective

---

## Foundation 0.1: Associative Memory Interface

**Source**: Definition 1 (Section 2) + Equation 3

**Graph Node**: `atlas_definitions/def-associative-memory`
**Equation Node**: `atlas_equations/eq-003-associative-memory-objective`

Equation 3 defines the optimization objective:
```
M* = argmin_M  L(M(K); V)
```

Where:
- `M` is a parametric function (the memory) mapping keys to values
- `K = {k_1, ..., k_n}` are context keys (input tokens projected)
- `V = {v_1, ..., v_n}` are context values (input tokens projected)
- `L` is a loss function measuring how well M(K) reconstructs V

**Pseudocode (abstract interface)**:

```
INTERFACE AssociativeMemory:
    """
    A parametric function M: K -> V that compresses context.
    This is NOT a separate module — it IS the parameters theta_M.
    The memory does not 'store' key-value pairs; it learns to
    reconstruct values from keys via gradient-based optimization.
    """

    PARAMETERS:
        theta_M : Tensor[]    -- memory weights (inner-loop optimized)

    METHODS:
        forward(K: Tensor[n, d_k]) -> Tensor[n, d_v]:
            """Apply memory to keys: M(K; theta_M)"""
            -- Concrete form depends on memory architecture
            -- Linear: theta_M @ K
            -- Deep MLP: W_L * sigma(... W_2 * sigma(W_1 * K))
            -- Polynomial: theta_M @ phi_p(K)

        compute_loss(K: Tensor[n, d_k], V: Tensor[n, d_v]) -> Scalar:
            """L(M(K; theta_M); V) — how well memory reconstructs values"""
            predictions = self.forward(K)
            RETURN loss_fn(predictions, V)

        update(K: Tensor[n, d_k], V: Tensor[n, d_v], lr: float) -> None:
            """One step of inner-loop optimization on theta_M"""
            loss = self.compute_loss(K, V)
            grad = gradient(loss, theta_M)
            theta_M = optimizer_step(theta_M, grad, lr)
            -- optimizer_step varies: GD, momentum, Newton-Schulz (Muon)
```

**Key constraint**: `forward()` is the ONLY external API. There is no separate
`memorize()`, `retrieve()`, or `forget()` — the memory learns through optimization,
and retrieval IS the forward pass.

---

## Foundation 0.2: Inner/Outer Loop Partition

**Source**: Appendix B, Definition in atlas_definitions/def-inner-outer-loop-partition

**Graph Node**: `atlas_definitions/def-inner-outer-loop-partition`

The Atlas family partitions all parameters into two groups with fundamentally
different optimization regimes:

```
PARTITION InnerOuterLoop:
    """
    Inner loop = test-time learning (memory adapts to new context)
    Outer loop = train-time learning (projections learn representations)

    This partition is NOT a mode switch (CS-10 compliance).
    Both loops operate during ALL processing — the difference
    is WHEN their parameters change and WHO optimizes them.
    """

    INNER_LOOP_PARAMS (theta_M):
        -- Memory MLP weights: {W_1, W_2, ..., W_{L_M}}
        -- Updated at TEST TIME via memory optimizer (GD/momentum/Muon)
        -- Updated PER CHUNK of context (not per epoch)
        -- Goal: minimize L(M(K); V) for current context

    OUTER_LOOP_PARAMS (theta_rest):
        -- Query/Key/Value projections: W_Q, W_K, W_V
        -- Short convolution layers
        -- Gating networks for alpha_t, eta_t, theta_t
        -- Non-memory MLP layers
        -- Layer norms
        -- Updated at TRAIN TIME via standard backpropagation
        -- Updated across the full training corpus
        -- Goal: learn good representations for the inner loop

    INVARIANT:
        -- Inner loop NEVER touches outer loop params
        -- Outer loop gradients flow THROUGH inner loop operations
        -- This is bilevel optimization: outer loop optimizes the
           initial conditions that make inner loop effective
```

---

## Foundation 0.3: Three-Axis Model Description

**Source**: Section 2, Table 1
**Graph Node**: `atlas_definitions/def-three-axis-model-description`

Atlas describes the entire design space using three axes.
Every model in the family is a specific choice on each axis.

```
DESCRIPTION ThreeAxisModel:
    """
    Three axes fully specify any model in the Atlas family.
    This collapses MIRAS's 4-choice grammar to 3 axes by
    folding the retention gate into the optimizer axis as
    'dynamic decay.'
    """

    AXIS 1 — Memory Architecture M(.):
        -- HOW the memory is structured
        -- Linear:     M(K) = theta_M @ K                    (capacity O(d_k))
        -- Deep MLP:   M(K) = W_L sigma(... W_1 K)           (capacity O(d_k * d_o * sum))
        -- Polynomial: M(K) = theta_M @ phi_p(K)             (capacity O(d_k^p))

    AXIS 2 — Attentional Bias l(.):
        -- WHAT objective guides memory updates
        -- Associative: l(k,v) = ||M(k) - v||^2              (reconstruct values)
        -- Surprise:    l(k,v) = -log p(v|M,k)               (novelty detection)
        -- Contrastive: l(k,v) = contrast(M(k), v)           (discriminative)
        -- None:        l = constant                          (no attentional modulation)

    AXIS 3 — Memory Optimizer:
        -- HOW memory parameters are updated
        -- GD:          theta -= lr * grad                    (first-order, O(1) state)
        -- Momentum:    m = beta*m + grad; theta -= lr*m      (smoothed, O(d) state)
        -- Newton-Schulz (Muon): orthogonalize grad, then step (second-order approx)
        -- Dynamic decay: alpha_t * theta + update            (retention folded in)

    THE SIX MODELS (Table 1 compositions):

    | Model            | M(.)       | l(.)         | Optimizer     | Characteristics Met |
    |------------------|------------|--------------|---------------|---------------------|
    | DLA              | Linear     | Associative  | GD            | 1/5 (Flexible Ctx)  |
    | SWLA             | Linear     | Associative  | GD+decay      | 2/5 (+Dynamic Decay)|
    | OmegaNet         | Linear     | Associative  | Momentum      | 2/5 (Flex+Optimal)  |
    | DeepTransformers | Deep MLP   | Associative  | GD+decay      | 3/5 (+Deep+Nonlin)  |
    | DoT              | Deep MLP   | Associative  | Momentum+decay| 4/5 (+Locally Opt)  |
    | Atlas            | Deep MLP   | Associative  | Muon+decay    | 5/5 (ALL)           |

    FIVE MEMORY CHARACTERISTICS (Table 1 binary columns):
        1. Dynamic Decay      -- alpha_t modulates retention per-token
        2. Deep Neural Memory -- multi-layer MLP, not single linear map
        3. Non-linear Capacity-- O(d_k^p) via polynomial features, not O(d_k)
        4. Locally Optimal    -- second-order optimizer (Muon), not just GD
        5. Flexible Context   -- sliding window of c tokens, not individual tokens
```

---

## Foundation 0.4: Five Memory Characteristics

**Source**: Table 1 (Section 1), Definition in `atlas_definitions/def-five-memory-characteristics`

```
DEFINITION FiveMemoryCharacteristics:
    """
    Binary properties that distinguish Atlas from all prior work.
    Each characteristic maps to a specific mechanism in the code.
    Atlas is the ONLY model that achieves all five.
    """

    1. DYNAMIC_DECAY:
        -- Mechanism: alpha_t = sigmoid(conv(x_t))
        -- What it does: per-token retention gating
        -- Without it: memory either remembers everything or decays uniformly
        -- Models with: SWLA, DeepTransformers, DoT, Atlas
        -- Models without: DLA, OmegaNet

    2. DEEP_NEURAL_MEMORY:
        -- Mechanism: L_M-layer MLP with nonlinear activations
        -- What it does: replaces linear map theta_M @ K with deep function
        -- Without it: capacity ceiling at O(d_k) per Proposition 1
        -- Models with: DeepTransformers, DoT, Atlas
        -- Models without: DLA, SWLA, OmegaNet

    3. NONLINEAR_CAPACITY:
        -- Mechanism: polynomial feature map phi_p(K)
        -- What it does: lifts capacity from O(d_k) to O(d_k^p)
        -- Without it: memorization quality bounded by key dimension
        -- Models with: Atlas (with polynomial features enabled)
        -- Equation: atlas_equations/eq-005-polynomial-feature-mapping

    4. LOCALLY_OPTIMAL:
        -- Mechanism: Muon optimizer (Newton-Schulz orthogonalization)
        -- What it does: approximates second-order optimization
        -- Without it: GD/momentum may converge to poor local optima
        -- Models with: DoT (momentum), Atlas (Muon)
        -- Models without: DLA, SWLA, DeepTransformers (plain GD)

    5. FLEXIBLE_CONTEXT:
        -- Mechanism: Omega Rule sliding window of c tokens
        -- What it does: memorizes CONTEXT WINDOWS, not individual tokens
        -- Without it: each token processed independently
        -- All Atlas-family models have this (it is the Omega Rule itself)
        -- Equation: atlas_equations/eq-009-omega-rule-update
```

---

## Tier 0 Audit Checklist

### Atlas IS Container (6 principles)

| # | Principle | Tier 0 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | Context memorization, not token memorization | PRESENT | Foundation 0.1: M maps context windows (K,V), Foundation 0.4 char #5 |
| 2 | Deep neural memory with non-linear capacity | PRESENT | Foundation 0.3 Axis 1 (Deep MLP), Foundation 0.4 chars #2,#3 |
| 3 | Locally optimal via advanced optimizers | PRESENT | Foundation 0.3 Axis 3 (Muon), Foundation 0.4 char #4 |
| 4 | Strict generalization of Transformers | DEFERRED | Proven by Eq 26, appears in Tier 2 (DeepTransformers variant) |
| 5 | Polynomial/exponential feature mappings | DEFERRED | Eq 5 implementation in Tier 1 |
| 6 | Composable building blocks | PRESENT | Foundation 0.3 shows 6 models from 3 axes |

### Atlas IS NOT Container (5 principles)

| # | Principle | Tier 0 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | NOT token-level memorization | COMPLIANT | Foundation 0.1: K is context window, not individual tokens |
| 2 | NOT limited to linear memory capacity | COMPLIANT | Foundation 0.3 Axis 1 includes Deep MLP + polynomial |
| 3 | NOT limited to simple GD | COMPLIANT | Foundation 0.3 Axis 3 goes GD -> momentum -> Muon |
| 4 | NOT replacement for Transformers | DEFERRED | Containment proof in Tier 2 |
| 5 | NOT a single architecture | COMPLIANT | Foundation 0.3 defines 6 models as compositions |

### NL IS Container (8 principles)

| # | Principle | Tier 0 Status | Notes |
|---|-----------|---------------|-------|
| 1 | New learning paradigm | COMPLIANT | Memory-as-optimization is the paradigm |
| 2 | Nested, multi-level, parallel optimization | DEFERRED | Atlas is ONE level; HOPE nests multiple Atlas instances |
| 3 | Each with its own context flow | COMPLIANT | Foundation 0.2: inner loop has its own context |
| 4 | Compressing their own context flow | COMPLIANT | Foundation 0.1: M* = argmin L(M(K); V) IS compression |
| 5 | In-context learning naturally emerges | COMPLIANT | Foundation 0.2: inner loop IS in-context learning |
| 6 | Optimizers are associative memory modules | COMPLIANT | Foundation 0.1: memory IS an optimizer objective |
| 7 | Self-modifying learning module | VALID GAP | GGD is HOPE-specific (Eq 88). Atlas modules are not self-modifying. |
| 8 | Continuum memory system | VALID GAP | CMS is HOPE-specific. Atlas provides the modules that CMS nests. |

### NL IS NOT Container (5 anti-principles)

| # | Anti-Principle | Tier 0 Status | Notes |
|---|---------------|---------------|-------|
| 1 | NOT single-level optimization | VALID GAP | Atlas IS single-level. It becomes multi-level when nested in HOPE CMS. |
| 2 | NOT shared/global context flow | COMPLIANT | Inner/outer have separate context |
| 3 | NOT static/fixed update rules | COMPLIANT | Dynamic decay, learnable gating |
| 4 | NOT discrete long/short-term memory | COMPLIANT | No discrete bins — memory is continuous parameters |
| 5 | NOT optimizers as just optimizers | COMPLIANT | Optimizer IS the memory update mechanism |

### Valid Gaps (not violations)

- **NL IS #2 (nesting)**: Atlas is a MODULE, not a full NL system. It becomes nested when placed inside HOPE's CMS.
- **NL IS #7 (self-modifying)**: GGD (Eq 88) is HOPE-specific. Atlas modules receive their update rules from outside.
- **NL IS #8 (CMS)**: CMS is the HOPE mechanism that nests Atlas modules at different frequencies.
- **NL IS NOT #1 (single-level)**: Atlas IS single-level by design — it's the building block that HOPE makes multi-level.

These gaps are architectural, not violations. Atlas + HOPE CMS = full NL compliance.

---

**Tier 0 Complete. Proceed to Tier 1: Shared Primitives.**
