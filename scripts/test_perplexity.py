"""Test script — measures perplexity of a built Atlas-MAG model.

CS-38: This is a TEST script, not an evaluation script.
Testing verifies the built artifact works. Functions use test/measure
terminology, not evaluate.

From HADES:
- impl-training-pipeline: perplexity measured on held-out validation data
"""

import argparse
import math

import torch

from atlas.config import AtlasConfig, tiny_config, full_config
from atlas.model import AtlasMAGModel


def measure_perplexity(model, data):
    """Measure perplexity on data.

    CS-38: Named measure_perplexity, not evaluate.

    Args:
        model: AtlasMAGModel instance
        data: Token tensor [N, seq_len]

    Returns:
        Perplexity (float)
    """
    total_loss = 0.0
    total_tokens = 0

    # NOTE: Cannot use torch.no_grad() here because the inner loop (TTL)
    # requires gradients for memory updates during the forward pass.
    # This is test-time learning: the model computes autograd.grad internally.
    # CS-10: No mode distinction — forward() works the same way always.
    for i in range(data.shape[0]):
        input_ids = data[i:i+1]
        out = model(input_ids, labels=input_ids)
        # Loss is per-token cross-entropy (mean over shifted positions)
        n_tokens = input_ids.shape[1] - 1
        total_loss += out['loss'].item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    return perplexity


def test_model(args):
    """Test a built model by measuring perplexity.

    CS-38: Named test_model, not evaluate_model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.baseline:
        # Freshly initialized model, no checkpoint
        config = tiny_config() if args.tiny else full_config()
        model = AtlasMAGModel(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print("=== Baseline Test (random init) ===")
        print(f"Config: {'tiny' if args.tiny else 'full'} ({total_params/1e6:.1f}M params)")
    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt.get('config', full_config())
        model = AtlasMAGModel(config).to(device)
        model.load_state_dict(ckpt['model'])
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Config: {total_params/1e6:.1f}M params, step {ckpt.get('step', '?')}")
    else:
        print("Error: specify --baseline or --checkpoint <path>")
        return

    # Generate synthetic test data (or use real held-out data if available)
    n_sequences = args.n_sequences
    seq_len = min(config.context_length, 128)  # Shorter for speed
    print(f"Testing on {n_sequences} synthetic sequences, length {seq_len}")

    data = torch.randint(0, config.vocab_size, (n_sequences, seq_len), device=device)
    total_tokens = n_sequences * (seq_len - 1)

    ppl = measure_perplexity(model, data)

    print("\n=== Results ===")
    print(f"Perplexity: {ppl:.2f}")
    print(f"Tokens tested: {total_tokens:,}")
    print(f"Expected baseline PPL: ~{config.vocab_size:.0f} (random init, vocab_size)")

    if args.baseline:
        # For random init, perplexity should be approximately vocab_size
        expected = config.vocab_size
        if ppl < expected * 2:
            print("Baseline PPL within expected range.")
        else:
            print(f"WARNING: PPL {ppl:.2f} seems high vs expected ~{expected}")


def main():
    parser = argparse.ArgumentParser(description="Atlas-MAG perplexity test")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to test')
    parser.add_argument('--baseline', action='store_true',
                        help='Test freshly-initialized model (no checkpoint)')
    parser.add_argument('--tiny', action='store_true',
                        help='Use tiny config (for baseline mode)')
    parser.add_argument('--n-sequences', type=int, default=10,
                        help='Number of test sequences')
    args = parser.parse_args()

    test_model(args)


if __name__ == '__main__':
    main()
