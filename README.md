# Atlas

Graph-derived pseudocode for the Atlas paper ([arXiv:2505.23735](https://arxiv.org/abs/2505.23735)) — *Learning to Optimally Memorize the Context at Test Time* by Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, and Mirrokni (Google Research).

## What This Is

This repository contains **algorithmic pseudocode** extracted from the Atlas paper using a semantic graph methodology. Every algorithm traces directly to numbered equations in the paper and has been audited against the paper's axiomatic constraints.

The pseudocode was generated from a HADES (ArangoDB) knowledge graph containing 60 equations, 9 definitions, 43 abstractions, and 2 axiom containers extracted from the paper. This is a one-way pipeline: **paper equations &rarr; graph &rarr; pseudocode**. The graph generates the pseudocode, not the other way around.

This repository serves as an exhibit for our paper on graph-driven paper-to-code methodology.

## Structure

```
docs/algorithms/
  00_tier0_foundations.md          # Associative memory interface, three-axis model, five memory characteristics
  01_tier1_shared_primitives.md    # QKV projections, polynomial features, Omega Rule, Newton-Schulz, dynamic decay
  02_tier2_variant_progression.md  # DLA -> SWLA -> OmegaNet -> DeepTransformers -> DoT -> Atlas
  03_tier3_parallelization.md      # Chunkwise Omega Rule, 4-phase Atlas parallel training
  04_tier4_compositions.md         # LMM, MAC, MAG, MAL composition patterns, full model stack
```

## Tier Progression

The pseudocode is organized in dependency order. Each tier builds on the previous:

| Tier | Name | Content | Equations |
|------|------|---------|-----------|
| 0 | Foundations | Core interfaces and axioms | Eq 3 (associative memory objective) |
| 1 | Shared Primitives | Reusable building blocks | Eqs 4-5, 8-11, 40, 42-43, 48, 52 |
| 2 | Variant Progression | 6 models, each adding one ingredient | Eqs 19-20, 25-27, 32-33, 45-58 |
| 3 | Parallelization | Chunkwise training algorithms | Eqs 15-16, 34-41 |
| 4 | Compositions | How memory integrates with attention | Eqs 17-18, SWA-Omega duality |

## The Atlas Variant Progression

Atlas is not a single model. It is a progression of 6 variants, each adding one ingredient:

```
DLA             = Linear attention + GD memory
SWLA            = DLA + Sliding Window Attention
OmegaNet        = SWLA + Omega Rule (momentum)
DeepTransformers = OmegaNet + Deep memory MLP
DoT             = DeepTransformers + Newton-Schulz orthogonalization
Atlas           = DoT + Polynomial feature expansion
```

## Cross-Paper Context

Atlas is part of a larger research program from the Mirrokni group:

- **Titans** (2501.00663) = WHERE memory goes (MAC/MAG/MAL placement)
- **MIRAS** (2504.13173) = WHAT memory does (4-axis design grammar)
- **Atlas** (2505.23735) = HOW MUCH capacity + HOW WELL optimized
- **TNT** (2511.07343) = HOW TO TRAIN (practical efficiency infrastructure)
- **Nested Learning** (2512.24695) = HOW MANY at what frequencies (CMS nesting)

## License

This repository contains algorithmic pseudocode derived from a published academic paper. The pseudocode is our original work; the underlying mathematical content is attributed to the paper authors.
