"""Outer loop optimization — constructs initial conditions for Atlas-MAG.

CS-38: This is a BUILD script, not a training script.
The outer loop constructs the initial conditions that enable the inner loop
to self-modify at test time. The model is BUILT, not trained.

From HADES:
- impl-training-pipeline: AdamW, lr=4e-4, cosine annealing, batch=0.5M tokens,
  weight_decay=0.1, dataset=FineWeb-Edu, tokenizer=T5 (32K vocab)
- def-inner-outer-loop-partition: outer loop optimizes ALL non-memory params
  + initial memory params via backprop
- CS-28: AdamW allowed only in outer loop
"""

import argparse
import json
import math
import os
import sys
import time
import itertools
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from atlas.config import AtlasConfig, tiny_config, full_config
from atlas.model import AtlasMAGModel


def probe_invariants(model, _step):
    """Probe architectural invariants during build (CS-30, CS-32).

    CS-30: Harmony is testable — verify gradient/state properties.
    CS-32: Observe then advance — probes run BEFORE step counter advances.

    Returns dict of probe measurements for logging.
    """
    raw_model = model.module if hasattr(model, 'module') else model
    probes = {}

    for i, block in enumerate(raw_model.blocks):
        mem = block.memory
        prefix = f"block_{i}"

        # Probe 1: S_init Frobenius norms per level (should be ~1.0 for unit sphere)
        for ell, S_init in enumerate(mem.S_inits):
            s_norm = S_init.data.pow(2).sum().sqrt().item()
            probes[f"{prefix}/level_{ell}/S_init_norm"] = s_norm

        # Probe 2: S_init gradient norms (should be non-zero after backward)
        for ell, S_init in enumerate(mem.S_inits):
            if S_init.grad is not None:
                g_norm = S_init.grad.pow(2).sum().sqrt().item()
                probes[f"{prefix}/level_{ell}/S_init_grad_norm"] = g_norm

        # Probe 3: Aggregation weight distribution (softmax normalized)
        w = torch.softmax(mem.agg_weights.data, dim=0)
        for ell in range(mem.k):
            probes[f"{prefix}/level_{ell}/agg_weight"] = w[ell].item()

    # Probe 4: Global gradient norm
    total_grad_norm = 0.0
    for p in raw_model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.pow(2).sum().item()
    probes["global_grad_norm"] = total_grad_norm ** 0.5

    # Probe 5: Gate centers — bias-only output (zero input) shows resting mu/eta
    for i, block in enumerate(raw_model.blocks):
        mem = block.memory
        prefix = f"block_{i}"
        mu_min = raw_model.config.lattice_mu_min
        mu_max = raw_model.config.lattice_mu_max
        eta_max = raw_model.config.lattice_eta_max
        with torch.no_grad():
            dev = next(mem.parameters()).device
            zero_in = torch.zeros(1, 1, mem.d, device=dev)
            mu_center = (mu_min + (mu_max - mu_min) * torch.sigmoid(mem.mu_gate(zero_in))).item()
            eta_center = (eta_max * torch.sigmoid(mem.eta_gate(zero_in))).item()
        probes[f"{prefix}/mu_center"] = mu_center
        probes[f"{prefix}/eta_center"] = eta_center

    return probes


def write_metrics(metrics_path, entry):
    """Append a metrics entry to JSONL file (one JSON object per line)."""
    with open(metrics_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def write_model_card(checkpoint_dir, config, total_params, metrics_history,
                     elapsed, tokens_processed):
    """Generate MODEL_CARD.md in the checkpoint directory."""
    final = metrics_history[-1] if metrics_history else {}

    card = f"""# Atlas-MAG Model Card

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Atlas-MAG (Memory As Gate) |
| Memory type | Lattice + CMS multi-rate orthogonal state recurrence |
| Parameters | {total_params/1e6:.1f}M |
| d_model | {config.d_model} |
| n_heads | {config.n_heads} |
| head_dim | {config.head_dim} |
| n_layers | {config.n_layers} |
| vocab_size | {config.vocab_size:,} |

## Memory Configuration

| Property | Value |
|----------|-------|
| Lattice slots (m) | {config.lattice_m if config.lattice_m > 0 else config.head_dim} |
| State decay (mu) | Learned (clamped [{config.lattice_mu_min}, {config.lattice_mu_max}], init {config.lattice_mu}) |
| Inner learning rate (eta) | Learned (clamped [0, {config.lattice_eta_max}], init {config.lattice_eta}) |
| Chunk size | {config.lattice_chunk_size} |
| CMS levels | {config.cms_levels} |
| CMS cadences | {config.cms_cadences} |
| BPTT length | {config.bptt_len} |

## Build Details

| Property | Value |
|----------|-------|
| Dataset | FineWeb-Edu (sample-10BT) |
| Tokenizer | T5 (32K vocab) |
| Context length | {config.context_length} |
| Batch size | {config.batch_size_tokens:,} tokens |
| Optimizer | AdamW (outer loop only) |
| Learning rate | {config.outer_lr} (cosine annealing) |
| Weight decay | {config.outer_weight_decay} |
| Total tokens | {tokens_processed:,} |
| Wall time | {elapsed:.1f}s ({elapsed/3600:.1f}h) |
| Throughput | {final.get('tok_per_sec', 0):.0f} tok/s |

## Final Metrics

| Metric | Value |
|--------|-------|
| Loss | {final.get('loss', 'N/A')} |
| Steps | {final.get('step', 'N/A')} |

## Architecture Notes

- **Inner loop**: Lattice orthogonal state recurrence (Karami et al. 2504.05646)
  runs during forward() — O(m*d) per token
- **Multi-rate updates**: CMS (Behrouz et al. 2512.24695, Eq 71/74) partitions
  memory into {config.cms_levels} frequency levels with cadences {config.cms_cadences}
- **Bilevel optimization**: Outer loop (AdamW) learns S_init + all non-memory params;
  inner loop updates state via orthogonal recurrence
- **No positional embedding**: Uses RoPE in the SWA attention branch
- **Weight tying**: LM head shares weights with token embedding
- **FP32 inner loop**: Memory update forced to FP32 via autocast(enabled=False)

## References

- Karami, Pascanu, Mirrokni. "Lattice: Learning to Efficiently Compress the Memory" (2504.05646)
- Behrouz, Zhong, Mirrokni. "Titans: Learning to Memorize at Test Time" (2501.00663)
- Behrouz et al. "It's All Connected" — CMS multi-rate updates (2512.24695)
"""
    card_path = os.path.join(checkpoint_dir, "MODEL_CARD.md")
    with open(card_path, 'w') as f:
        f.write(card)
    print(f"Saved model card: {card_path}")


def get_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Cosine annealing learning rate schedule (impl-training-pipeline)."""
    if step >= total_steps:
        return min_lr
    progress = step / total_steps
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def create_data_iterator(config, rank, world_size, device):
    """Create streaming data iterator from FineWeb-Edu.

    impl-training-pipeline: FineWeb-Edu, T5 tokenizer (32K vocab), streaming.
    Packs text into fixed-length sequences of context_length tokens.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Shard across DDP workers
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)

    seq_len = config.context_length
    token_buffer = []

    for example in dataset:
        text = example.get("text", "")
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(tokens)

        # Yield complete sequences
        while len(token_buffer) >= seq_len:
            seq = token_buffer[:seq_len]
            token_buffer = token_buffer[seq_len:]
            yield torch.tensor(seq, dtype=torch.long, device=device)


def build_step(model, batch, optimizer, grad_accum_steps, use_amp=False,
               probe_fn=None):
    """Execute one outer loop build step with gradient accumulation.

    CS-38: Named build_step, not train_step.
    CS-28: AdamW is the outer loop optimizer (allowed).
    CS-32: probe_fn runs AFTER backward, BEFORE optimizer.zero_grad().
    """
    total_loss = 0.0
    for micro_batch in batch:
        input_ids = micro_batch.unsqueeze(0)  # [1, seq_len]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            out = model(input_ids, labels=input_ids)
            loss = out['loss'] / grad_accum_steps
        loss.backward()
        total_loss += out['loss'].item()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # CS-32: Probes observe BEFORE optimizer clears gradients
    probe_result = None
    if probe_fn is not None:
        probe_result = probe_fn()

    optimizer.step()
    optimizer.zero_grad()

    return total_loss / grad_accum_steps, probe_result


def run_build(args):
    """Main build loop.

    CS-38: Named run_build, not train.
    """
    # DDP setup
    is_ddp = 'RANK' in os.environ
    if is_ddp:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Config
    if args.dry_run:
        config = tiny_config()
        total_steps = 1
        if rank == 0:
            print("=== DRY RUN: tiny config, 1 step, synthetic data ===")
    else:
        config = full_config()
        if args.context_length:
            config.context_length = args.context_length
        if args.batch_size:
            config.batch_size_tokens = args.batch_size
        total_steps = args.max_steps
        if rank == 0:
            print("=== Atlas-MAG Build ===")
            print("Dataset: FineWeb-Edu (streaming)")
            print(f"Context length: {config.context_length}")
            print(f"Batch size: {config.batch_size_tokens:,} tokens")
            if args.bf16:
                print("Mixed precision: BF16")

    # Model
    model = AtlasMAGModel(config).to(device)

    if args.gradient_checkpoint:
        model.enable_gradient_checkpointing()
        if rank == 0:
            print("Gradient checkpointing: enabled")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {total_params/1e6:.1f}M parameters")
        print(f"Device: {device}, World size: {world_size}")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Outer loop optimizer (impl-training-pipeline: AdamW, CS-28 compliant)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.outer_lr,
        weight_decay=config.outer_weight_decay,
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if is_ddp else model
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']

    # Gradient accumulation: accumulate over micro-batches to hit target batch size
    # Each micro-batch = 1 sequence of context_length tokens
    # Target = batch_size_tokens per step across all workers
    seqs_per_step_per_worker = max(1, config.batch_size_tokens // (config.context_length * world_size))
    grad_accum_steps = seqs_per_step_per_worker

    if rank == 0:
        print(f"Gradient accumulation: {grad_accum_steps} sequences/step/worker "
              f"({grad_accum_steps * config.context_length * world_size:,} tokens/step)")
        print(f"Max steps: {total_steps}")
        print()

    # Data iterator
    if args.dry_run:
        data_iter = iter([torch.randint(0, config.vocab_size, (config.context_length,), device=device)
                          for _ in range(10)])
    else:
        data_iter = create_data_iterator(config, rank, world_size, device)

    # Build loop
    tokens_processed = 0
    t0 = time.time()
    metrics_history = []

    # Create checkpoint dir early so metrics.jsonl is available from step 0
    if args.checkpoint_dir and rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_path = os.path.join(args.checkpoint_dir, "metrics.jsonl") if args.checkpoint_dir else None

    # Probe frequency: every N steps (avoid overhead on every step)
    probe_every = max(1, args.probe_every)

    for step in range(start_step, start_step + total_steps):
        # Learning rate schedule
        lr = get_cosine_lr(step, total_steps + start_step, config.outer_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Collect micro-batches for gradient accumulation
        micro_batches = []
        for _ in range(grad_accum_steps):
            try:
                seq = next(data_iter)
                micro_batches.append(seq)
            except StopIteration:
                if rank == 0:
                    print("Data exhausted, stopping.")
                break

        if not micro_batches:
            break

        # CS-32: Probe function captures gradients BEFORE optimizer.zero_grad()
        should_probe = (rank == 0) and (step % probe_every == 0)
        probe_fn = (lambda _step=step: probe_invariants(model, _step)) if should_probe else None

        loss, probe_result = build_step(model, micro_batches, optimizer,
                                        len(micro_batches), use_amp=args.bf16,
                                        probe_fn=probe_fn)
        step_tokens = sum(mb.numel() for mb in micro_batches) * world_size
        tokens_processed += step_tokens
        elapsed = time.time() - t0

        if rank == 0:
            tok_per_sec = tokens_processed / max(elapsed, 1e-6)

            entry = {
                'step': step,
                'loss': round(loss, 6),
                'lr': lr,
                'tokens': tokens_processed,
                'tok_per_sec': round(tok_per_sec, 1),
                'elapsed_s': round(elapsed, 2),
            }

            # Invariant probes (CS-30: harmony is testable)
            if probe_result is not None:
                entry['probes'] = probe_result

            metrics_history.append(entry)

            # Write to metrics file
            if metrics_path:
                write_metrics(metrics_path, entry)

            # Console output
            print(f"step {step:>6d} | loss {loss:.4f} | lr {lr:.2e} | "
                  f"tokens {tokens_processed:>12,d} | {tok_per_sec:.0f} tok/s")
            sys.stdout.flush()

        # Checkpoint saving
        if args.checkpoint_dir and (step + 1) % args.save_every == 0 and rank == 0:
            raw_model = model.module if is_ddp else model
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step+1}.pt")
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step + 1,
                'config': config,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            sys.stdout.flush()

    if is_ddp:
        dist.destroy_process_group()

    if rank == 0:
        elapsed = time.time() - t0
        print(f"\nBuild finished. {tokens_processed:,} tokens in {elapsed:.1f}s")

        # Generate model card
        if args.checkpoint_dir:
            total_params = sum(p.numel() for p in (model.module if is_ddp else model).parameters())
            write_model_card(args.checkpoint_dir, config, total_params,
                             metrics_history, elapsed, tokens_processed)


def main():
    parser = argparse.ArgumentParser(description="Atlas-MAG outer loop build")
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 1 step on tiny config with synthetic data')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Maximum build steps')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--gradient-checkpoint', action='store_true',
                        help='Enable gradient checkpointing for memory savings')
    parser.add_argument('--context-length', type=int, default=None,
                        help='Override context length (default: use config value)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 mixed precision (recommended for A6000)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size_tokens (default: use config value)')
    parser.add_argument('--probe-every', type=int, default=10,
                        help='Run invariant probes every N steps (default: 10)')
    args = parser.parse_args()

    run_build(args)


if __name__ == '__main__':
    main()
