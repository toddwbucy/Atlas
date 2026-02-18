# Atlas

A component-by-component implementation of the Atlas paper, built from graph-derived pseudocode and used to analyze the training dynamics and deployment constraints of test-time learning architectures.

> **Atlas: Learning to Optimally Memorize the Context at Test Time**
> Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, Mirrokni (Google Research)
> [arXiv:2505.23735](https://arxiv.org/abs/2505.23735)

**Companion checkpoint** (43M model, 85.9% NIAH): [r3d91ll/Atlas-MAG_OmegaRule on HuggingFace](https://huggingface.co/r3d91ll/Atlas-MAG_OmegaRule)

**First implementation** (archived): [toddwbucy/Atlas-MAG_OmegaRule](https://github.com/toddwbucy/Atlas-MAG_OmegaRule)

## What This Repository Contains

This is not a trained model. This is the **dissection**.

After building a working 43M Atlas-MAG model (the companion repo above), we rebuilt it component by component to understand where the compute cost, precision requirements, and scaling constraints actually come from. Each component was added incrementally — polynomial features, then the Lattice inner loop, then CMS frequency nesting — measuring training dynamics at each step.

What we found:

- **Polynomial features** square values that go subnormal at FP16. Quantize the model and the memory module produces infinity. Not degraded performance — *infinity*.
- **Scaling to 120M parameters** produces sawtooth loss divergence. The training dynamics become unstable in ways that standard remedies (learning rate warmup, gradient clipping) do not fix.
- **Every serving platform** assumes you can quantize, shard, and batch. This model says no to all three.

The full analysis is in [`docs/reports/`](docs/reports/).

## Quick Start

```bash
git clone https://github.com/toddwbucy/Atlas.git
cd Atlas
pip install -r requirements.txt

# Run tests (46 tests, ~2 seconds)
PYTHONPATH=src python -m pytest tests/ -v
```

## Four Memory Exhibits

The implementation progresses through four memory architectures, each building on the previous:

| Exhibit | Memory Type | What It Adds | Key Finding |
|---------|------------|--------------|-------------|
| `memory.py` | Base AtlasMemory | Polynomial features + Omega Rule + Newton-Schulz | Capacity scales as O(d_k^2) but requires FP32 |
| `memory_omega.py` | MemoryOmega | Momentum recurrence, sliding context window | Muon momentum enables parallelizable training |
| `memory_lattice.py` | MemoryLattice | Orthogonal state recurrence (Lattice inner loop) | Quarter-memory beats full TTT memory |
| `memory_mlp.py` | MemoryMLP | Full MLP memory + CMS frequency nesting | Multi-rate updates at 4 frequency levels |

## Architecture

```
Input -> Embedding -> [MAGBlock x N] -> RMSNorm -> LM Head -> Output

MAGBlock:
    x --+--> [Sliding Window Attention] --> attn_out
        |                                      |
        +--> [Deep Polynomial Memory]  --> mem_out
                                               |
        output = x + attn_out * sigmoid(mem_out)
```

## Project Structure

```
Atlas/
├── docs/
│   ├── algorithms/                    # Graph-derived pseudocode (5 tiers)
│   │   ├── 00_tier0_foundations.md
│   │   ├── 01_tier1_shared_primitives.md
│   │   ├── 02_tier2_variant_progression.md
│   │   ├── 03_tier3_parallelization.md
│   │   └── 04_tier4_compositions.md
│   └── reports/
│       ├── atlas_graph_audit.pdf      # Validation report
│       └── atlas_graph_audit.tex
├── src/atlas/
│   ├── config.py                      # AtlasConfig dataclass (all params HADES-traced)
│   ├── model.py                       # AtlasMAG full model
│   ├── blocks.py                      # MAGBlock (attention + memory gating)
│   ├── attention.py                   # Sliding Window Attention with RoPE
│   ├── primitives.py                  # Polynomial features, Newton-Schulz, Omega loss
│   ├── memory.py                      # Base AtlasMemory (Exhibit 1)
│   ├── memory_omega.py                # MemoryOmega with momentum (Exhibit 2)
│   ├── memory_lattice.py              # MemoryLattice with OSR (Exhibit 3)
│   └── memory_mlp.py                  # MemoryMLP with CMS nesting (Exhibit 4)
├── scripts/
│   ├── build.py                       # DDP training with probes and metrics
│   └── test_perplexity.py             # Perplexity evaluation
├── tests/                             # 46 tests
├── AGENT_BUILD_PROMPT.md              # Prompt used to build this from HADES graph
└── requirements.txt
```

## Pseudocode Tiers

The code was generated from graph-derived pseudocode, organized in dependency order:

| Tier | Name | Content | Equations |
|------|------|---------|-----------|
| 0 | Foundations | Core interfaces and axioms | Eq 3 (associative memory objective) |
| 1 | Shared Primitives | Reusable building blocks | Eqs 4-5, 8-11, 40, 42-43, 48, 52 |
| 2 | Variant Progression | 6 models, each adding one ingredient | Eqs 19-20, 25-27, 32-33, 45-58 |
| 3 | Parallelization | Chunkwise training algorithms | Eqs 15-16, 34-41 |
| 4 | Compositions | How memory integrates with attention | Eqs 17-18, SWA-Omega duality |

## Cross-Paper Context

Atlas is part of a larger research program from the Mirrokni group:

- **Titans** (2501.00663) = WHERE memory goes (MAC/MAG/MAL placement)
- **MIRAS** (2504.13173) = WHAT memory does (4-axis design grammar)
- **Atlas** (2505.23735) = HOW MUCH capacity + HOW WELL optimized
- **TNT** (2511.07343) = HOW TO TRAIN (practical efficiency infrastructure)
- **Lattice** (2504.05646) = HOW TO COMPRESS (orthogonal state recurrence)
- **Nested Learning** (2512.24695) = HOW MANY at what frequencies (CMS nesting)

## Citation

```bibtex
@article{behrouz2025atlas,
  title={Atlas: Learning to Optimally Memorize the Context at Test Time},
  author={Behrouz, Ali and Li, Yingcong and Kacham, Praneeth and Daliri, Poria and Deng, Zhihao and Zhong, Peilin and Razaviyayn, Meisam and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2505.23735},
  year={2025}
}
```

## License

MIT
