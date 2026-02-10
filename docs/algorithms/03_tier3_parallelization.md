# Tier 3: Chunkwise Parallelization

> **Purpose**: Transform the sequential per-token algorithms of Tier 2 into
> GPU-efficient chunk-parallel form. This is the critical bridge between
> mathematical correctness (Tier 2) and practical trainability.
>
> **Depends on**: Tier 1 (all primitives), Tier 2 (Atlas variant specifically)
>
> **Graph Sources**: atlas_equations (Eqs 15-16, 34-41)
> atlas_abstractions (atlas-parallel-training)
> **Cross-paper**: TNT (Titans iNside Titans) extends this with hierarchical memory

---

## The Parallelization Problem

Tier 2's Atlas algorithm (Variant 2.6) is sequential: each token's memory update
depends on the previous token's state. This creates a chain dependency:

```
M_0 -> M_1 -> M_2 -> ... -> M_N     (sequential, O(N) time)
```

GPUs need parallel work. The insight from Section 5.1: break the sequence into
chunks and parallelize WITHIN each chunk while maintaining recurrence ACROSS chunks.

```
[Chunk 0: t=0..b-1] -> [Chunk 1: t=b..2b-1] -> ... -> [Chunk T/b]
 ^^ parallel ^^          ^^ parallel ^^
```

---

## Algorithm 3.1: Chunkwise Omega Rule

**Source**: Eq 15 (chunk-wise omega update), Eq 16 (intra-chunk parallel)
**Graph Nodes**: `atlas_equations/eq-015-chunkwise-omega-update`, `atlas_equations/eq-016-intrachunk-parallel`

This applies to ALL variants (DLA through Atlas). The key trick: gradients within
a chunk are computed w.r.t. the LAST state of the previous chunk (M_{t'}),
not the per-token state M_{t-1}.

```
ALGORITHM chunkwise_omega(x: Tensor[N, d_in],
                           W_Q, W_K, W_V,
                           M_init: AssociativeMemory,
                           chunk_size: int = b,
                           c: int,                    -- window size
                           variant_config              -- which variant
                           ) -> Tensor[N, d_v]:
    """
    Chunkwise parallel Omega Rule (Eqs 15-16).

    Partition sequence into chunks of size b.
    Within each chunk: compute all gradients in parallel
    Across chunks: sequential memory state updates

    Approximation: gradients use M_{t'} (chunk boundary) not M_{t-1} (per-token).
    When b=1: exact (reduces to Tier 2 sequential form).
    When b>1: approximation that enables parallelism.

    From TNT paper: this approximation is empirically negligible
    and enables 17x speedup.
    """
    REQUIRE N % chunk_size == 0 or handle padding
    num_chunks = ceil(N / chunk_size)

    Q, K, V = qkv_project(x, W_Q, W_K, W_V)
    -- Apply feature mapping based on variant
    K, Q = apply_features(K, Q, variant_config)

    M = M_init
    all_outputs = []

    FOR chunk_idx IN 0..num_chunks-1:
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_len = chunk_end - chunk_start

        -- Extract chunk tokens
        K_chunk = K[chunk_start:chunk_end]   -- [b, d_k]
        V_chunk = V[chunk_start:chunk_end]   -- [b, d_v]
        Q_chunk = Q[chunk_start:chunk_end]   -- [b, d_k]

        -- ====== PARALLEL WITHIN CHUNK ======

        -- All gradients computed w.r.t. M (chunk boundary state)
        -- NOT M_{t-1} (this is the approximation)
        grads = parallel_compute_gradients(M, K_chunk, V_chunk, c, gamma)
        -- grads: [b, *theta_shape]  -- one gradient per position

        -- Apply sliding window mask M_s (Eq 16)
        -- M_s[i,j] = 1 if j in [i-c+1, i], else 0
        -- This masks which past positions contribute to each gradient
        masked_grads = apply_window_mask(grads, c)

        -- Compute dynamic decay for chunk
        alphas = compute_dynamic_decay(x[chunk_start:chunk_end], alpha_conv)
        -- alphas: [b, 1]

        -- ====== SEQUENTIAL ACROSS CHUNK ======

        -- Update memory state for this chunk
        -- (Sequential because M_t depends on M_{t-1})
        chunk_outputs = []
        FOR t_local IN 0..chunk_len-1:
            M.theta_M = alphas[t_local] * M.theta_M - eta * masked_grads[t_local]
            y_t = M.forward(Q_chunk[t_local])
            chunk_outputs.append(y_t)

        all_outputs.extend(chunk_outputs)

    RETURN stack(all_outputs)
```

---

## Algorithm 3.2: Atlas Parallel Training (Full)

**Source**: Eqs 34-41 (Section 5.1)
**Graph Nodes**: `atlas_equations/eq-034` through `eq-041`
**Abstraction**: `atlas_abstractions/atlas-parallel-training`

The full Atlas parallelization is more sophisticated than generic chunkwise Omega.
It exploits the fact that momentum (S_t) is independent of memory (M_t) when
gradients use chunk-boundary state.

```
ALGORITHM atlas_parallel_training(x: Tensor[N, d_in],
                                    W_Q, W_K, W_V,
                                    M_init: GatedMemoryMLP,
                                    S_init: Tensor,
                                    alpha_conv, theta_conv,
                                    eta_net,
                                    chunk_size: int = b,
                                    c: int,
                                    phi: PolynomialFeatures,
                                    ns_iterations: int = 5
                                    ) -> Tensor[N, d_v]:
    """
    Atlas parallel training (Section 5.1, Eqs 34-41).

    Four-phase pipeline per chunk:
        Phase 1: Compute all gradients in parallel       (Eq 35)
        Phase 2: Accumulate momentum in parallel         (Eq 39)
        Phase 3: Newton-Schulz5 in parallel             (Eq 40)
        Phase 4: Sequential memory update                (Eq 41)

    KEY INSIGHT (Eq 39): S_t depends only on S_0 (chunk start), gradients,
    and decay products — NOT on M_t. This makes Phases 1-3 fully parallel.
    Only Phase 4 is sequential (b steps per chunk, not N).

    COMPOSABLE PARAMETERS:
        alpha_t: per-token memory decay      (outer loop, learned)
        theta_t: per-token momentum decay    (outer loop, learned)
        eta_t:   per-token learning rate     (outer loop, learned)
        c:       sliding window size         (hyperparameter)
        k:       Newton-Schulz iterations    (test-time compute budget)
    """
    num_chunks = ceil(N / chunk_size)
    Q, K, V = qkv_project(x, W_Q, W_K, W_V)
    K = polynomial_features(K, phi.p, phi.coefficients)
    Q = polynomial_features(Q, phi.p, phi.coefficients)

    M = M_init
    S = S_init
    all_outputs = []

    FOR chunk_idx IN 0..num_chunks-1:
        cs = chunk_idx * chunk_size
        ce = min(cs + chunk_size, N)
        K_c, V_c, Q_c = K[cs:ce], V[cs:ce], Q[cs:ce]
        x_c = x[cs:ce]

        -- Compute per-token parameters (all outer-loop, all parallel)
        alpha = compute_dynamic_decay(x_c, alpha_conv)    -- [b]
        theta = compute_momentum_decay(x_c, theta_conv)   -- [b]
        eta = eta_net(x_c)                                 -- [b]

        -- ═══════════════════════════════════════════
        -- PHASE 1: Parallel gradient computation (Eq 35)
        -- ═══════════════════════════════════════════
        -- All gradients w.r.t. M (frozen chunk boundary state)
        -- G[t] = grad ||M(phi(k_t)) - v_t||^2, for t in chunk
        G = []
        PARALLEL FOR t IN 0..b-1:
            -- Omega Rule gradient for position t
            start = max(0, (cs + t) - c + 1) - cs  -- window start within chunk
            end = t + 1
            K_win = K_c[max(0,start):end]
            V_win = V_c[max(0,start):end]
            gamma = gamma_net(x_c[max(0,start):end])
            G[t] = gradient(omega_rule_loss(M, K_win, V_win, gamma), M.theta_M)
        -- G: [b, *theta_shape]

        -- ═══════════════════════════════════════════
        -- PHASE 2: Parallel momentum accumulation (Eq 39)
        -- ═══════════════════════════════════════════
        -- S_t = beta_t * S_0 - Theta (*) E (*) G
        -- where beta_t = prod(theta_1..theta_t), E = diag(eta), Theta = decay matrix
        --
        -- This is a matrix multiply, NOT a sequential scan.
        -- All S_t computed in one shot.

        -- Build decay products
        beta = cumprod(theta)                   -- [b], beta_t = prod theta_1..theta_t
        -- Build weighted gradient matrix
        PARALLEL FOR t IN 0..b-1:
            -- S_t = beta[t] * S_chunk_start
            --      - sum_{j=0}^{t} (prod theta_{j+1}..theta_t) * eta[j] * G[j]
            weighted_sum = zeros_like(S)
            FOR j IN 0..t:
                decay_factor = prod(theta[j+1:t+1]) if j < t else 1.0
                weighted_sum += decay_factor * eta[j] * G[j]
            S_all[t] = beta[t] * S - weighted_sum
        -- S_all: [b, *theta_shape]

        -- ═══════════════════════════════════════════
        -- PHASE 3: Parallel Newton-Schulz5 (Eq 40)
        -- ═══════════════════════════════════════════
        PARALLEL FOR t IN 0..b-1:
            S_orth[t] = newton_schulz5(S_all[t], ns_iterations)
        -- S_orth: [b, *theta_shape]

        -- ═══════════════════════════════════════════
        -- PHASE 4: Sequential memory update (Eq 41)
        -- ═══════════════════════════════════════════
        chunk_outputs = []
        FOR t IN 0..b-1:
            M.theta_M = alpha[t] * M.theta_M - eta[t] * S_orth[t]
            y_t = M.forward(Q_c[t])
            chunk_outputs.append(y_t)

        -- Update momentum state for next chunk
        S = S_all[b-1]

        all_outputs.extend(chunk_outputs)

    RETURN stack(all_outputs)
```

---

## Complexity Analysis

| Phase | Work | Parallelism | Wall Time |
|-------|------|-------------|-----------|
| Phase 1: Gradients | O(b * c * d) | Full (b independent) | O(c * d) |
| Phase 2: Momentum | O(b^2 * d) | Full (matrix multiply) | O(b * d) with matmul |
| Phase 3: Newton-Schulz | O(b * k * d^2) | Full (b independent) | O(k * d^2) |
| Phase 4: Memory update | O(b * d) | Sequential | O(b * d) |

**Total wall time per chunk**: O(c*d + b*d + k*d^2 + b*d)
**vs Tier 2 sequential**: O(b * (c*d + d^2))

The speedup comes from Phases 1-3 being parallelizable. Phase 4 is O(b) sequential
steps but each step is just a multiply-add (no gradient computation).

**From TNT paper**: chunkwise parallel gives 17.37x speedup over fully sequential.

---

## Connection to TNT (Titans iNside Titans)

TNT extends this parallelization with three additional innovations:

1. **Hierarchical memory**: 1 global + N local memories (not in this pseudocode)
2. **Q-K Projection**: Aligns compression and retrieval domains
3. **Two-stage training**: Stage 1 uses small chunks (speed), Stage 2 uses large chunks (quality)

These are compositional extensions on TOP of Algorithm 3.2. They don't change
the core parallelization — they change how memory is structured (hierarchical)
and how training is scheduled (two-stage).

**TNT pseudocode will be a separate document** (`tnt_algorithms.md`) that imports
Algorithm 3.2 as a subroutine.

---

## Tier 3 Audit Checklist

### Atlas IS Container (6 principles)

| # | Principle | Tier 3 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | Context memorization | PRESERVED | Sliding window mask M_s (Eq 16) preserves c-token context |
| 2 | Deep neural memory | PRESERVED | Memory architecture unchanged from Tier 2 |
| 3 | Locally optimal | PRESERVED | Newton-Schulz5 in Phase 3, applied to all positions |
| 4 | Strict generalization | PRESERVED | Linear special case still contains Transformers |
| 5 | Polynomial features | PRESERVED | Applied before chunking (independent of parallelism) |
| 6 | Composable | ENHANCED | Chunk size b is a new composable parameter |

### Atlas IS NOT Container (5 principles)

| # | Principle | Tier 3 Status | Evidence |
|---|-----------|---------------|----------|
| 1 | NOT token-level | COMPLIANT | Window mask ensures c-token context within chunks |
| 2 | NOT linear-only | COMPLIANT | Deep MLP and polynomial features preserved |
| 3 | NOT simple GD | COMPLIANT | Muon preserved in Phase 3 |
| 4 | NOT replacement | COMPLIANT | Parallelization is implementation, not architecture change |
| 5 | NOT single architecture | COMPLIANT | Algorithm 3.1 is variant-agnostic |

### Approximation Audit

The ONLY approximation introduced in Tier 3:
- Gradients in Eq 35 use M_{t'} (chunk boundary) instead of M_{t-1} (per-token)
- When chunk_size = 1: EXACT (recovers Tier 2)
- When chunk_size > 1: approximation grows with chunk size
- TNT paper validates: negligible loss degradation for practical chunk sizes

This approximation does NOT violate any IS/IS NOT principle — it's a numerical
approximation, not a conceptual change.

### NL IS/IS NOT

Same valid gaps as Tiers 0-2. Parallelization is within a single level.

---

**Tier 3 Complete. Proceed to Tier 4: Compositions (MAC, MAG, MAL, MAD).**
