from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

import transformers
import transformers.modeling_utils as modeling_utils

if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        """
        Minimal stub to satisfy older model code that imports this symbol.
        It is only used when pruning attention heads; in this project we never call it.
        """
        # Keep the signature compatible; simply return remaining heads and identity indices.
        heads = set(heads) - already_pruned_heads
        import torch
        index = torch.arange(0, n_heads * head_size, dtype=torch.long)
        return heads, index

    # Inject the function into transformers.modeling_utils so dynamic modules can import it
    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

class TeacherNT500M(nn.Module):
    """
    Teacher wrapper around a local NT-500M checkpoint.
    Expected directory structure:
        src/hf/nucleotide-transformer-v2-500m-multi-species/
            - config.json
            - pytorch_model.bin (or safetensors)
            - tokenizer.json / merges / vocab
    """
    def __init__(self, model_dir: str = "../src/hf/nucleotide-transformer-v2-500m-multi-species", device: Optional[str] = None):
        super().__init__()
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
        # Ensure we load the MLM head to get logits
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
        self.model.config.output_hidden_states = True
        self.model.eval()

        if device is not None:
            self.to(device)

        # Cache some handy properties
        self.vocab_size = self.model.get_input_embeddings().num_embeddings
        self.mask_token_id = self.tokenizer.mask_token_id
        # Teacher dims (embedding and hidden are typically equal)
        self.teacher_embed_dim = self.model.get_input_embeddings().embedding_dim
        # Try to read hidden size; fall back to embed dim if missing
        self.teacher_hidden_dim = getattr(self.model.config, "hidden_size", self.teacher_embed_dim)

    @torch.no_grad()
    def mask_with_positions(self, input_ids: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
        """
        Replace token ids at masked_positions with [MASK].
        input_ids: [B, L]
        masked_positions: [B, M] with valid indices in [0, L).
        """
        x = input_ids.clone()
        B, M = masked_positions.shape
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(B, M)
        x[batch_idx, masked_positions] = self.mask_token_id
        return x

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        masked_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
            - input_embeds: [B, L, Dt]   (teacher token embeddings for the original input)
            - last_hidden:  [B, L, Dt]   (teacher last hidden for the original input)
            - masked_logits:[B, M, V]    (teacher logits at masked positions; M=masked_positions.size(1))
            - misc: vocabulary size and mask token id
        Notes:
            * We run two passes:
              (1) original input -> to get input embeddings and last hidden (for representation KD)
              (2) masked input    -> to get logits at masked positions (for KL KD)
        """
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # (1) Original sequence -> embeddings + last hidden
        # Input embeddings lookup
        input_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L, Dt]

        out_orig = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        # hidden_states: tuple(length = n_layers+1), we take the last one
        last_hidden = out_orig.hidden_states[-1]  # [B, L, Dt]

        # (2) Masked logits (only if masked_positions provided)
        if masked_positions is not None:
            masked_ids = self.mask_with_positions(input_ids, masked_positions)
            out_mask = self.model(
                input_ids=masked_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            logits_full = out_mask.logits  # [B, L, V]

            # Gather logits at masked positions -> [B, M, V]
            B, M = masked_positions.shape
            V = logits_full.size(-1)
            idx = masked_positions.unsqueeze(-1).expand(B, M, V)
            masked_logits = logits_full.gather(dim=1, index=idx).contiguous()
        else:
            masked_logits = None

        return {
            "input_embeds": input_embeds,              # [B, L, Dt]
            "last_hidden": last_hidden,                # [B, L, Dt]
            "masked_logits": masked_logits,            # [B, M, V] or None
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "teacher_embed_dim": self.teacher_embed_dim,
            "teacher_hidden_dim": self.teacher_hidden_dim,
        }
