"""AtlasMAGModel — full language model with MAG blocks.

From HADES:
- model-atlas-mag: Atlas MAG with SWA, ~54M params at full config
- architectural-backbone: embedding -> MAG blocks -> LM head
- impl-positional-encoding: NO absolute positional embeddings (RoPE is in SWA)
- impl-training-pipeline: vocab_size=32000 (T5 tokenizer)

CS-10: No train/eval mode distinction.
CS-13: No "training" in class/method names.
CS-18: forward() is the only public API.
CS-38: Build not train, test not eval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import AtlasConfig
from .blocks import MAGBlock


class AtlasMAGModel(nn.Module):
    """Full Atlas-MAG language model.

    Architecture:
        Token embedding (NO position embedding — RoPE in SWA)
        -> N x MAGBlock
        -> RMSNorm
        -> LM head (tied weights with embedding)
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = False

        # Token embedding — NO absolute positional embedding (impl-positional-encoding)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Stacked MAG blocks
        self.blocks = nn.ModuleList([
            MAGBlock(config) for _ in range(config.n_layers)
        ])

        # Final norm before LM head
        self.final_norm = nn.RMSNorm(config.d_model)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        Modules with _skip_model_init=True (e.g. gate modules with calibrated
        bias) are left untouched to preserve their custom initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_skip_model_init', False):
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to trade compute for memory.

        Re-computes block activations during backward instead of storing them.
        Uses use_reentrant=False for compatibility with the inner-loop autograd
        in AtlasMemory.
        """
        self._gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """Forward pass: input_ids -> logits (+ optional loss).

        CS-18: forward() is the only public API.

        Args:
            input_ids: Token IDs [B, T]
            labels: Optional target IDs [B, T] for computing cross-entropy loss

        Returns:
            Dict with 'logits' [B, T, vocab_size] and optionally 'loss' (scalar)
        """
        # Embed tokens
        x = self.token_emb(input_ids)  # [B, T, d_model]

        # Pass through MAG blocks
        for block in self.blocks:
            if self._gradient_checkpointing and x.requires_grad:
                x = torch_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        result = {'logits': logits}

        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            result['loss'] = loss

        return result
