# Atlas Agent Build Prompt

## Objective

Build a complete Atlas-MAG language model with the Omega Rule, from scratch, using ONLY the HADES knowledge graph as your specification source. The target is a ~54M parameter model that can be instantiated, run forward, compute loss, and backpropagate gradients through the bilevel optimization (inner loop memory updates + outer loop backpropagation).

**You are building the MAG (Memory As Gate) variant of the Atlas model family.**

## Specification Source: HADES Graph (ONLY)

Your ONLY source of truth for what to build is the HADES graph database. You must query it to discover:
- What equations define the math
- What definitions establish the interfaces
- What abstractions describe the architecture
- What code smells constrain the implementation
- What axioms govern the design

**You may NOT read any files outside your workspace directory.** No peeking at other implementations, no reading archived code, no consulting pseudocode documents. Everything comes from HADES queries.

### Workspace Boundary (CRITICAL)

Your workspace is **`/home/todd/testing/atlas/`**. ALL files you create — source code, tests, configs, scripts — MUST be written inside this directory. Do NOT write files anywhere else on the filesystem. Specifically:
- Do NOT write to `/home/todd/olympus/` or any subdirectory
- Do NOT write to `/home/todd/testing/` root (only `/home/todd/testing/atlas/`)
- If a path in this prompt references `/home/todd/olympus/NestedLearning/Atlas/`, translate it to `/home/todd/testing/atlas/`

You may READ from HADES (the graph database) — that is your specification source. But all OUTPUT goes to your workspace only.

### How to Query HADES

```bash
# CRITICAL: Always set the database
export HADES_DATABASE=NL

# AQL queries (structured)
hades db aql 'FOR doc IN atlas_equations RETURN {key: doc._key, label: doc.label, description: doc.description}'

# Semantic search (natural language)
hades db query "omega rule memory update" --paper 2505.23735 2>/dev/null

# Get a specific document
hades db aql 'FOR doc IN atlas_equations FILTER doc._key == "eq-009-omega-rule" RETURN doc'
```

**Important**: `hades db query` sends progress to stderr. Use `2>/dev/null` when piping JSON.

### Collections You Must Query

These collections contain the Atlas specification:

| Collection | Count | What It Contains |
|------------|-------|-----------------|
| `atlas_equations` | 60 | All equations (numbered + unnumbered). The math. |
| `atlas_definitions` | 10 | Formal definitions: associative memory, inner/outer loop partition, three-axis model, five memory characteristics |
| `atlas_abstractions` | 51 | Architectural concepts: model-atlas-mag, impl-newton-schulz5, impl-ttl-hyperparameters, impl-polynomial-features, impl-positional-encoding, impl-inner-loop-memory-management, atlas-parallel-training, model variants, implementation notes |
| `atlas_axioms` | 2 | IS container (6 principles) and IS NOT container (5 anti-principles) |
| `atlas_lineage` | 7 | How concepts evolved across papers |
| `nl_code_smells` | 38 | Enforcement rules (CS-01 through CS-38). Code MUST NOT violate these. |
| `nl_axioms` | 2 | Project-wide IS/IS NOT containers |
| `nl_probe_patterns` | 5 | Invariant probes -- physics checks |
| `paper_edges` | 63 | Cross-paper connections (Atlas to Titans to HOPE) |
| `titans_abstractions` | 14 | MAG pattern originated in Titans -- query this for MAG composition rules |
| `build_runs` | 1+ | Build run logs. Read the CHARTER node for schema. **You MUST write a build run node here.** |

### Recommended Query Sequence

1. **Start with axioms** -- understand what this model IS and IS NOT
2. **Read definitions** -- understand the formal interfaces (associative memory, inner/outer partition)
3. **Read the MAG abstraction** -- atlas_abstractions/model-atlas-mag tells you the architecture
4. **Read key equations** -- Eqs 9 (Omega Rule), 32-33 (Atlas update), 40 (Newton-Schulz5), 42-43 (deep MLP), 57-58 (implementation-ready Atlas)
5. **Read TTL hyperparameters** -- atlas_abstractions/impl-ttl-hyperparameters has the inner loop parameter specification (theta, alpha, eta, gamma, NS coefficients)
6. **Read polynomial features spec** -- atlas_abstractions/impl-polynomial-features specifies the CORRECT construction (outer-product, NOT element-wise powers). This is critical for O(d_k^2) capacity.
7. **Read positional encoding spec** -- atlas_abstractions/impl-positional-encoding specifies RoPE for the SWA branch (paper is silent on this, but the working model used RoPE). Do NOT use absolute positional embeddings.
8. **Read inner loop memory management** -- atlas_abstractions/impl-inner-loop-memory-management specifies the chunkwise detachment pattern to avoid O(T) memory growth. CRITICAL for real sequences.
9. **Read the outer loop pipeline** -- atlas_abstractions/impl-training-pipeline has the outer loop specification (AdamW, lr, schedule, batch size, weight decay, dataset). This is what `scripts/build.py` implements.
10. **Read code smells** -- know what NOT to do before you start coding. Pay special attention to CS-38 (build not train, test not eval) for script naming.
11. **Cross-reference Titans** -- MAG pattern came from Titans, check titans_abstractions

## Target Model Specification

Build the MAG variant with these parameters:

### Full Config (~54M parameters)
```
d_model = 768
n_heads = 12
head_dim = 64  (d_model / n_heads)
n_layers = 6
vocab_size = 32000
memory_expansion = 4  (hidden_dim = key_dim * expansion)
poly_degree = 2  (polynomial features for O(d_k^2) capacity)
omega_context_window = 256
ns_iterations = 5  (Newton-Schulz5)
swa_window_size = 512
```

### Tiny Config (for CPU testing, under 1 second)
```
d_model = 64
n_heads = 4
head_dim = 16
n_layers = 2
vocab_size = 256
memory_expansion = 4
poly_degree = 2
omega_context_window = 8
ns_iterations = 5
swa_window_size = 16
```

## Target Directory Structure

Write all code under /home/todd/testing/atlas/:

```
atlas/
  src/atlas/
    __init__.py           # Exports
    config.py             # AtlasConfig dataclass + tiny_config()
    primitives.py         # Pure functions: polynomial_features, newton_schulz5, omega_loss
    memory.py             # AtlasMemory (deep MLP with polynomial features)
    attention.py          # SlidingWindowAttention
    blocks.py             # MAGBlock (memory + attention + gating)
    model.py              # AtlasMAGModel (full model)
  scripts/
    build.py              # Outer loop optimization — constructs initial conditions (DDP, torchrun compatible)
    test_perplexity.py    # Tests the built model — measures perplexity on held-out data
  tests/
    conftest.py           # Tiny fixtures
    test_primitives.py    # Tests for pure functions
    test_memory.py        # Tests for memory module
    test_blocks.py        # Tests for MAG block
    test_model.py         # Tests for full model
```

## Verification Criteria

Your build is complete when ALL of the following pass:

### 1. Smoke Test
```python
from atlas.model import AtlasMAGModel
from atlas.config import tiny_config
import torch

model = AtlasMAGModel(tiny_config())
input_ids = torch.randint(0, 256, (2, 32))
out = model(input_ids, labels=input_ids)
print(f"Loss: {out['loss'].item():.4f}")
print(f"Logits shape: {out['logits'].shape}")  # [2, 32, 256]
out['loss'].backward()
print("Backward OK")
```

### 2. Parameter Count
```python
from atlas.config import full_config
model = AtlasMAGModel(full_config())
total = sum(p.numel() for p in model.parameters())
assert 40_000_000 < total < 70_000_000, f"Expected ~54M params, got {total/1e6:.1f}M"
```

### 3. Bilevel Gradient Flow
After loss.backward(), memory initialization parameters must have non-None gradients. The inner loop (TTL) modifies memory at test time; the outer loop learns the initial conditions via backprop.

### 4. All Tests Pass
```bash
cd /home/todd/testing/atlas
# First time: create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch pytest datasets transformers

# Run tests
PYTHONPATH=src pytest tests/ -v
```

### 5. Code Smell Compliance
After building, verify against the code smell rules you queried from nl_code_smells. Key ones to check:
- CS-10: No model.train() / model.eval() mode distinction
- CS-13: The word "training" should not appear in class/method names
- CS-18: forward() is the only public API method on blocks
- CS-31: The memory layer is indivisible (not decomposed into sub-modules that can be separately called)
- CS-38: Build not train, test not eval — script names, function names, docstrings

### 6. Build Script Runs (CPU)
The outer loop script (`scripts/build.py`) must:
- Accept `--dry-run` flag that runs 1 step on tiny config and exits (verifies the full pipeline without real data)
- Dry run completes without error in under 30 seconds on CPU

### 7. Test Script Runs (CPU)
The test script (`scripts/test_perplexity.py`) must:
- Accept a checkpoint path and compute perplexity on synthetic data when no real data is available
- Run against a freshly-initialized model (no checkpoint) to produce a baseline perplexity
- **Use `--tiny` for baseline testing** — full config may OOM on a single GPU during test because the inner loop requires gradients even for inference

### 8. GPU Dry Run
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/build.py --dry-run
```
Must complete without CUDA errors. This verifies device placement and CUDA compatibility.

### 9. DDP Build (First 10 Steps)
```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src torchrun --nproc_per_node=2 scripts/build.py \
    --checkpoint-dir runs/atlas_54m --max-steps 1000 --save-every 500
```
Must run for at least 10 steps without OOM or NaN. Loss should decrease. If this fails, enter the Iterative Failure Recovery loop.

## Build and Test Scripts

### scripts/build.py — Outer Loop Optimization

This script constructs the model's initial conditions via the outer loop. Query `atlas_abstractions/impl-training-pipeline` for the full specification. Key requirements:

**Outer loop spec (from graph):**
- Optimizer: AdamW (this is the OUTER loop — CS-28 allows frequency-aware optimizers)
- Learning rate: 4e-4 base, cosine annealing schedule
- Batch size: 0.5M tokens
- Weight decay: 0.1
- Context length: 4096 tokens per sequence
- Dataset: FineWeb-Edu (streaming via HuggingFace datasets)

**Script requirements:**
- **DDP compatible**: Must work with `torchrun --nproc_per_node=N scripts/build.py`
- **Streaming data**: Use HuggingFace `datasets` library with streaming=True (no full download)
- **Checkpoint saving**: Save model state + optimizer state + step count at configurable intervals
- **Checkpoint resuming**: `--resume <path>` to continue from a saved checkpoint
- **Dry run mode**: `--dry-run` runs 1 step on tiny config with synthetic data, validates the full pipeline
- **Logging**: Print loss, learning rate, tokens processed, throughput (tok/s) at each step
- **Gradient checkpointing**: Enable via `--gradient-checkpoint` flag for memory savings
- **CS-38 compliance**: No function named `train()`. Use `build_step()`, `run_build()`, or similar.

**GPU layout**: GPU 0+1 = A6000 49GB (build via DDP), GPU 2 = RTX 2000 Ada 16GB (HADES, do not use)

Example invocations:
```bash
# Dry run (CPU, tiny model, 1 step)
PYTHONPATH=src python scripts/build.py --dry-run

# Single GPU build
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/build.py --checkpoint-dir runs/atlas_54m

# DDP build (2x A6000)
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src torchrun --nproc_per_node=2 scripts/build.py --checkpoint-dir runs/atlas_54m

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src torchrun --nproc_per_node=2 scripts/build.py --resume runs/atlas_54m/step_1000.pt
```

### scripts/test_perplexity.py — Model Testing

Tests a built model by measuring perplexity on held-out data. Query `atlas_abstractions/impl-training-pipeline` for dataset details.

**Script requirements:**
- Accept `--checkpoint <path>` to load a built model
- Accept `--baseline` to test a freshly-initialized model (no checkpoint)
- Compute perplexity on held-out data (or synthetic data if no real data available)
- Report: perplexity, number of tokens tested, model config summary
- **CS-38 compliance**: No function named `evaluate()`. Use `test_model()`, `measure_perplexity()`, or similar.

Example invocations:
```bash
# Baseline perplexity (random init)
PYTHONPATH=src python scripts/test_perplexity.py --baseline

# Test a checkpoint
PYTHONPATH=src python scripts/test_perplexity.py --checkpoint runs/atlas_54m/step_1000.pt
```

## Constraints

1. **Graph-only specification**: Every design decision must trace to a HADES query result. If you cannot find it in the graph, query more deeply before inventing.

2. **No old code access**: Do NOT read files from anywhere outside your workspace (`/home/todd/testing/atlas/`). Specifically forbidden:
   - /home/todd/olympus/ (entire olympus tree — hands off)
   - /home/todd/olympus/Acheron/ (archived implementations)
   - /home/todd/olympus/Atlas-MAG_OmegaRule/ (old repo)
   - /home/todd/olympus/NestedLearning/ (the real project directory)
   - Any other Atlas or HOPE implementation

3. **No pseudocode files**: Do NOT read any .md files outside your workspace. Your source is the HADES graph itself, not human summaries.

4. **Code smell compliance**: Query nl_code_smells and obey all 38 rules (CS-01 through CS-38). The graph is the immune system -- code that violates it is rejected.

5. **Axiom compliance**: Query atlas_axioms and nl_axioms. Your model must comport with the IS container and not violate the IS NOT container. Atlas has valid gaps for NL IS #2 (nesting), #7 (self-modifying), #8 (CMS) -- these are architectural by design because Atlas is a single-level module.

6. **Paper equation traceability**: Every function you write should be traceable to an equation in atlas_equations. Comment the equation reference (e.g., # Eq 9: Omega Rule).

## When Code Fails: The Graph Feedback Loop

The HADES graph is not just a specification — it is a diagnostic tool. When your code fails tests or produces incorrect results:

1. **Do NOT guess the fix.** Go back to the graph.
2. **Re-query the relevant equation(s)** for the failing component. Read the full document — descriptions often contain implementation constraints you may have missed on first read.
3. **Query code smells** to check if you've introduced a violation. A code smell violation often manifests as a subtle bug, not an obvious error.
4. **Query abstractions** for implementation notes. Nodes like `impl-ttl-hyperparameters`, `impl-newton-schulz5`, `alg-atlas-complete` contain detailed pseudocode steps and parameter specifications.
5. **Check cross-paper references.** If an equation links to a Titans equation, query that too — the sibling paper may clarify ambiguity.
6. **Compare your implementation against the graph's specification** line by line. The graph is the source of truth.
7. **Fix the code to match the graph**, then re-run tests.

**The graph is the immune system.** If your code doesn't work, the answer is in the graph — not in your intuition. Every successful fix should trace to a specific graph query that revealed the discrepancy.

### Graph Gap Alerts

If after thorough graph queries you still cannot find something you need, **you MUST emit a structured alert**. Do NOT silently invent an answer. Do NOT guess and move on. The graph is a living document and gaps get fixed in real time — but only if we know about them.

**Emit this exact format:**

```
🔴 GRAPH GAP ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT I NEED:    [what you're looking for]
WHY I NEED IT:  [what component/function requires this]
WHAT I QUERIED:
  - [collection/key or semantic query 1]
  - [collection/key or semantic query 2]
  - [collection/key or semantic query 3]
WHAT I FOUND:   [closest match, or "nothing relevant"]
SEVERITY:       [CRITICAL = cannot proceed | MODERATE = can proceed with assumption | MINOR = cosmetic]
MY ASSUMPTION:  [if MODERATE/MINOR, what you'll assume until corrected]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Severity rules:**
- **CRITICAL**: Stop building that component. Move to the next independent component and come back later. Do NOT write code based on a guess for critical gaps.
- **MODERATE**: Proceed with your stated assumption, but mark the assumption in the code with `# GRAPH GAP: <description>` so it can be found and corrected later.
- **MINOR**: Proceed with best judgment. Note it but don't block progress.

Every gap alert is valuable — it tells us exactly where the graph needs to grow. This is how the graph stays alive.

## Build Approach

1. **Query first, code second.** Spend significant time understanding the graph before writing any code. The graph contains 60 equations, 10 definitions, 51 abstractions -- read them.

2. **Build bottom-up.** Start with pure functions (primitives), then memory module, then attention, then MAG block, then full model. Each layer builds on the previous.

3. **Test each layer.** Write tests as you go. Do not build the full model and then debug -- validate each component.

4. **The inner loop runs in forward.** This is the critical insight: memory parameters are updated DURING the forward pass (test-time learning / TTL). The outer loop learns the initial memory parameters and all non-memory parameters via standard backpropagation. Query the definitions for inner-outer-loop-partition to understand this.

5. **MAG = Memory As Gate.** The MAG composition runs attention and memory in parallel, then uses memory output to gate attention output: output = x + attn_out * sigmoid(mem_out). Query atlas_abstractions/model-atlas-mag and titans_abstractions for the pattern.

## Build Run Logging (REQUIRED)

You MUST create a build run node in the `build_runs` collection to document your entire build process. This is not optional — it is how the graph learns from builds and how code smells get discovered.

### Step 1: Read the Charter

Before creating your build run, query the charter to understand the schema:

```bash
export HADES_DATABASE=NL
hades db aql 'FOR doc IN build_runs FILTER doc._key == "CHARTER" RETURN doc.node_schema'
```

### Step 2: Create Your Build Run Node (at start)

At the BEGINNING of your session, before you start writing code, create your build run node:

```bash
hades db aql '
INSERT {
  _key: "run-YYYY-MM-DD-HHMM",
  paper: "2505.23735",
  model_variant: "atlas-mag",
  config_used: { type: "full_config", d_model: 768, n_heads: 12, n_layers: 6, params_target: "54M" },
  agent_session: "<your session identifier>",
  timestamp_start: "<ISO 8601 now>",
  timestamp_end: null,
  axiom_basis: "nl_axioms/NL_IS",
  validated_against: "nl_axioms/NL_IS_NOT",
  timeline: [],
  files_produced: [],
  outcome: null,
  code_smell_candidates: [],
  graph_gaps_discovered: []
} INTO build_runs RETURN NEW._key
'
```

Use today's date and current time for the `_key`. This node is your build's living record.

### Step 3: Update Timeline During Build

As you work, accumulate timeline events. You don't need to update the graph after every single action — batch updates at key milestones:

**Milestone 1: After initial graph queries** (before coding)
**Milestone 2: After all source files are created**
**Milestone 3: After tests pass**
**Milestone 4: After each GPU execution attempt** (success or failure)
**Milestone 5: After each failure recovery iteration**

Update with:
```bash
hades db update build_runs run-YYYY-MM-DD-HHMM --json '{
  "timeline": [ ...accumulated events... ]
}'
```

### Step 4: Finalize at End

When your build is complete (success or final failure), update the node with:
- `timestamp_end`: ISO 8601 completion time
- `outcome`: final status, test counts, loss, throughput
- `files_produced`: all files with equation traceability
- `code_smell_candidates`: any failure patterns you observed that might become enforcement rules
- `graph_gaps_discovered`: any GRAPH GAP ALERTs you emitted

### Timeline Event Types

Record these as you go:

```json
// When you query the graph
{ "type": "graph_query", "collection": "atlas_equations", "key_or_query": "eq-009-omega-rule",
  "what_found": "Omega Rule: weighted L2 regression loss", "used_for": "memory.py omega_loss" }

// When you create a file
{ "type": "file_created", "path": "src/atlas/memory.py", "line_count": 211,
  "traces_to_equations": ["eq-009", "eq-042", "eq-043", "eq-057", "eq-058"] }

// When tests run
{ "type": "test_result", "test_file": "tests/test_memory.py", "passed": 8, "failed": 0, "duration_seconds": 1.2 }

// When you execute on GPU
{ "type": "execution", "command": "torchrun --nproc_per_node=2 scripts/build.py ...",
  "target": "DDP build", "result_summary": "10 steps completed, loss 8.23 -> 7.91" }

// When something fails
{ "type": "failure", "error_type": "OOM", "component": "polynomial_features",
  "traceback_summary": "Tried to allocate 45.75 GiB at primitives.py:34",
  "tensor_shapes": { "input": "[61, 4096, 12, 64]", "attempted_output": "[61, 4096, 12, 64, 64]" } }

// When you diagnose via graph
{ "type": "diagnosis", "graph_queries_made": ["atlas_abstractions/impl-polynomial-features"],
  "what_found": "MEMORY SCALING WARNING: eager computation of outer product",
  "root_cause": "Pre-computed polynomial features for entire [B,T,H,d] tensor instead of per-window" }

// When you fix based on graph
{ "type": "fix", "file_modified": "src/atlas/memory.py", "iteration": 1,
  "change_description": "Moved polynomial feature computation inside inner loop, per-window",
  "graph_source": "atlas_abstractions/impl-polynomial-features" }

// When the graph is missing something
{ "type": "graph_gap", "what_needed": "inference-time create_graph behavior",
  "why_needed": "test_perplexity.py OOMs because create_graph=True not needed for inference",
  "what_queried": ["impl-inner-loop-memory-management", "impl-training-pipeline"],
  "severity": "MODERATE", "assumption": "Use create_graph=False when no outer loop gradient needed" }
```

### Code Smell Candidates

When you observe a failure pattern that could help future builds, record it:

```json
{
  "pattern": "Eager polynomial feature computation OOMs at scale",
  "severity": "CRITICAL",
  "failure_context": "Pre-computing [B,T,H,d,d] outer product for full sequence allocates 45.75 GiB",
  "fix_applied": "Compute lazily per-window inside inner loop",
  "promoted_to": null
}
```

The `promoted_to` field stays null — a human reviewer will later decide whether to promote this to a formal `nl_code_smells/CS-N` node. When they do, the code smell will have a `discovered_in` field pointing back to your build run.

**This is how the graph grows.** Your build failures become tomorrow's enforcement rules.

## Dataset & Dependencies

### Python Dependencies

Your venv MUST include these packages. Install them before running any code:

```bash
cd /home/todd/testing/atlas
python3 -m venv venv
source venv/bin/activate
pip install torch pytest datasets transformers
```

| Package | Why |
|---------|-----|
| `torch` | Model framework |
| `pytest` | Test runner |
| `datasets` | HuggingFace streaming data (FineWeb-Edu) |
| `transformers` | Tokenizer (T5, 32K vocab) |

### Dataset: FineWeb-Edu (Streaming)

The outer loop build uses FineWeb-Edu as the context source. Query `atlas_abstractions/impl-training-pipeline` for the full spec. Key details:

- **Dataset**: `HuggingFaceFW/fineweb-edu` (subset `sample-10BT`)
- **Tokenizer**: `google-t5/t5-base` (32,000 token vocabulary — matches `config.vocab_size`)
- **Streaming**: `load_dataset(..., streaming=True)` — no full download, infinite iterator
- **Packing**: Concatenate tokenized text into fixed-length sequences of `config.context_length` tokens
- **DDP sharding**: When using multiple GPUs, shard the stream: `dataset.shard(num_shards=world_size, index=rank)`

```python
# Pattern for streaming data (impl-training-pipeline)
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

# Pack into fixed-length sequences
token_buffer = []
for example in dataset:
    tokens = tokenizer.encode(example["text"], add_special_tokens=False)
    token_buffer.extend(tokens)
    while len(token_buffer) >= context_length:
        yield torch.tensor(token_buffer[:context_length], dtype=torch.long, device=device)
        token_buffer = token_buffer[context_length:]
```

### GPU Layout

| GPU | Type | VRAM | Purpose |
|-----|------|------|---------|
| 0 | A6000 | 49 GB | Build (DDP rank 0) |
| 1 | A6000 | 49 GB | Build (DDP rank 1) |
| 2 | RTX 2000 Ada | 16 GB | HADES embedding service — **DO NOT USE** |

**Always use `CUDA_VISIBLE_DEVICES=0,1`** when launching builds to exclude GPU 2.

## Execution Phase: Build After Tests Pass

After all tests pass and verification criteria are met, you MUST execute the build pipeline on GPU. This is not optional — the model must actually run on real hardware with real data.

### Execution Sequence

1. **All tests pass** (`PYTHONPATH=src pytest tests/ -v` — 0 failures)
2. **Dry run passes** (`PYTHONPATH=src python scripts/build.py --dry-run` — completes without error)
3. **GPU dry run passes** (`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/build.py --dry-run` — verifies CUDA compatibility)
4. **Launch DDP build** on both A6000 GPUs:

```bash
cd /home/todd/testing/atlas
source venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src torchrun --nproc_per_node=2 scripts/build.py \
    --checkpoint-dir runs/atlas_54m \
    --max-steps 1000 \
    --save-every 500
```

5. **Monitor the first 10 steps.** If loss is decreasing and no errors occur, the build is working. You do not need to wait for all 1000 steps — verifying the first 10 is sufficient.

6. **If the build fails (OOM, NaN, crash, import error):** Enter the Iterative Failure Recovery loop (next section).

### What "Working Build" Means

After step 5, report:
- Loss at step 0 and step 10 (should be decreasing)
- Throughput in tokens/second
- GPU memory usage (if available via `torch.cuda.max_memory_allocated()`)
- Whether any NaN/Inf values appeared

## Iterative Failure Recovery (CRITICAL)

When GPU execution fails — OOM, NaN loss, CUDA errors, shape mismatches, or any runtime error — you MUST use the graph to diagnose and fix. **Do NOT guess.** The graph contains the answer.

### Recovery Protocol

```
FAILURE DETECTED
    │
    ▼
1. READ THE FULL TRACEBACK
    │  - What function failed?
    │  - What tensor shape or memory allocation?
    │  - What line of your code?
    │
    ▼
2. MAP FAILURE TO GRAPH COMPONENT
    │  - OOM → query impl-polynomial-features, impl-inner-loop-memory-management
    │  - NaN → query equations for numerical stability constraints
    │  - Shape mismatch → query equations for exact tensor dimensions
    │  - Mode error → query nl_code_smells for CS-10 violations
    │
    ▼
3. RE-QUERY THE GRAPH (deeper this time)
    │  - Read the FULL document, not just the summary
    │  - Check the "common_mistakes" field if present
    │  - Check cross-paper references (Titans, HOPE patterns)
    │  - Query: hades db query "<error description>" --paper 2505.23735 2>/dev/null
    │
    ▼
4. IDENTIFY THE DISCREPANCY
    │  - What does your code do vs what the graph specifies?
    │  - Is there a memory scaling issue the graph warns about?
    │  - Did you miss a constraint in an abstraction node?
    │
    ▼
5. FIX THE CODE TO MATCH THE GRAPH
    │  - Make the minimal change needed
    │  - Comment the graph source: # Fixed per atlas_abstractions/<key>
    │
    ▼
6. RE-RUN TESTS
    │  - PYTHONPATH=src pytest tests/ -v (must still pass)
    │  - Then re-run the failing execution step
    │
    ▼
7. IF STILL FAILING → go back to step 1
    │  Each iteration MUST include at least one new graph query
    │  Do NOT retry the same fix twice
    │
    ▼
8. IF GRAPH DOESN'T HAVE THE ANSWER → emit GRAPH GAP ALERT
    │  Then make your best assumption and proceed
```

### Common GPU Failures and Graph References

| Failure | Graph Source to Query | What to Look For |
|---------|----------------------|------------------|
| OOM during forward pass | `impl-polynomial-features`, `impl-inner-loop-memory-management` | Memory scaling warnings, lazy computation pattern, chunkwise detachment |
| OOM during backward pass | `impl-inner-loop-memory-management` | `create_graph=True` scope — should only be True within chunks, not across entire sequence |
| NaN loss | Equations for Omega Rule (Eq 9), NS5 (Eq 40) | Numerical guards (epsilon, clamping), division by zero in normalization |
| Loss not decreasing | `impl-ttl-hyperparameters` | Learning rates (eta, alpha, theta), NS5 coefficient values |
| CUDA error / device mismatch | `impl-training-pipeline` | DDP setup, `find_unused_parameters=True`, device placement |
| Shape mismatch | `atlas_equations` for the specific equation | Exact tensor dimensions — d_k vs d_v vs d_model |
| Very slow throughput | `impl-inner-loop-memory-management` | Chunk size, omega window size, unnecessary recomputation |

### Maximum Recovery Iterations

If after **5 iterations** of the recovery loop you cannot resolve the failure:
1. Emit a GRAPH GAP ALERT with all 5 attempts documented
2. Describe what you tried and what the traceback shows
3. Stop and report — do not continue guessing

## What Success Looks Like

A working Atlas-MAG model that builds, tests, and runs on GPU:

**Code Quality:**
- Instantiates with ~54M parameters (full config) or tiny config for testing
- Runs forward: input_ids to logits (and optionally computes cross-entropy loss)
- Runs backward: gradients flow through both inner and outer loop parameters
- Implements the Omega Rule (Eq 9) for memory updates within the forward pass
- Uses Newton-Schulz5 (Eq 40) to orthogonalize momentum before memory updates
- Uses polynomial features (Eq 5) to increase memory capacity to O(d_k^2)
- Combines attention and memory via MAG gating
- Passes all tests
- Complies with all 38 code smell rules (including CS-38: build not train)
- Every function traces to a graph equation

**Execution:**
- `scripts/build.py --dry-run` completes successfully on CPU
- `scripts/build.py --dry-run` completes successfully on GPU (CUDA_VISIBLE_DEVICES=0)
- `scripts/test_perplexity.py --baseline --tiny` reports baseline perplexity
- DDP build runs for 10+ steps on 2x A6000 GPUs without OOM or NaN
- Loss decreases over the first 10 steps
- If any execution fails, the failure was diagnosed and fixed via graph queries (documented)

**Build Run Log:**
- A `build_runs` node exists in HADES with your run's `_key`
- Timeline captures graph queries, files created, test results, failures, diagnoses, and fixes
- `files_produced` links every source file to the equations it implements
- `code_smell_candidates` documents any failure patterns observed
- `graph_gaps_discovered` documents any specification gaps found
