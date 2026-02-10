from typing import Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Small utility building blocks
# ----------------------------

class RMSNorm(nn.Module):
    """RMSNorm with stable epsilon. Used instead of LayerNorm for efficiency."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class DepthwiseConv1d(nn.Module):
    """Depthwise 1D conv with SAME length output."""
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        # Use 'same' padding to keep length == L for any kernel_size (PyTorch >= 1.10/2.x).
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            groups=d_model, padding='same', bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L] -> conv -> [B, L, D]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        # Safety clamp for older PyTorch or edge cases: crop/trim if shapes drift.
        if y.size(1) != x.size(1):
            # Keep the leftmost L tokens to exactly match the input length.
            y = y[:, :x.size(1), :]
        return y


# ----------------------------
# Sparse Mixture-of-Experts (MoE) feed-forward layer
# References:
#   - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017)
#   - Lepikhin et al., "GShard" (NeurIPS 2020)
#   - Fedus et al., "Switch Transformers" (arXiv 2021 / JMLR 2022)
# Notes:
#   - Token-level Top-k routing with a simple load-balancing auxiliary loss.
#   - This implementation uses a Python loop over experts (OK for research-scale E<=16).
# ----------------------------

class ExpertMLP(nn.Module):
    """A single expert: 2-layer MLP with SiLU activation."""
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(d_hidden, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TopKRouter(nn.Module):
    """Linear router producing routing logits over experts."""
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        self.proj = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D] -> logits: [N, E]
        return self.proj(x)


class MoEFeedForward(nn.Module):
    """
    Token-level sparse MoE FFN.
    Input/Output: [B, L, D] -> [B, L, D]
    Also returns an auxiliary load-balancing loss (scalar).
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        ffn_mult: float = 4.0,
        dropout: float = 0.0,
        moe_init_zero: bool = True,
    ):
        super().__init__()
        assert n_experts >= 1, "n_experts must be >= 1"
        assert top_k in (1, 2), "This simple implementation supports top_k in {1,2}"
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        d_hidden = int(round(ffn_mult * d_model))
        self.router = TopKRouter(d_model=d_model, n_experts=n_experts)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_hidden, dropout=dropout) for _ in range(n_experts)])

        # Function-preserving init for dense->MoE upcycling:
        # Start MoE residual branch near-zero so loading a dense checkpoint keeps behavior stable.
        if moe_init_zero:
            nn.init.zeros_(self.router.proj.weight)
            for exp in self.experts:
                nn.init.zeros_(exp.fc2.weight)
                if exp.fc2.bias is not None:
                    nn.init.zeros_(exp.fc2.bias)

    @staticmethod
    def _load_balance_loss(probs: torch.Tensor, top1_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
        """
        Switch-style load balancing loss.
        probs:    [N, E] softmax over experts.
        top1_idx: [N]    top-1 expert index for each token.
        """
        # Importance: mean probability mass assigned to each expert.
        importance = probs.mean(dim=0)  # [E]
        # Load: fraction of tokens routed (top-1) to each expert.
        load = torch.zeros((n_experts,), device=probs.device, dtype=probs.dtype)
        load.scatter_add_(0, top1_idx, torch.ones_like(top1_idx, dtype=probs.dtype))
        load = load / max(1.0, float(top1_idx.numel()))
        # E * sum(importance * load)
        return (importance * load).sum() * float(n_experts)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        x: [B, L, D]
        attn_mask: [B, L] with 1 for valid, 0 for pad (optional)
        Returns:
          y: [B, L, D]
          aux_loss: scalar tensor
        """
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)  # [N, D]
        if attn_mask is not None:
            m = attn_mask.reshape(B * L).bool()
            valid_idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        else:
            valid_idx = torch.arange(B * L, device=x.device)

        if valid_idx.numel() == 0:
            return torch.zeros_like(x), x.new_zeros(())

        x_valid = x_flat.index_select(0, valid_idx)  # [Nv, D]
        logits = self.router(x_valid)                # [Nv, E]
        probs = torch.softmax(logits, dim=-1)        # [Nv, E]

        # Top-k routing (indices + scores)
        topk_scores, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # [Nv, k]
        # Normalize weights across selected experts to sum to 1.
        topk_w = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)  # [Nv, k]

        # Auxiliary load-balancing loss uses top-1 decisions.
        aux_loss = self._load_balance_loss(probs=probs, top1_idx=topk_idx[:, 0], n_experts=self.n_experts)

        # Dispatch to experts
        y_valid = torch.zeros_like(x_valid)  # [Nv, D]
        for e in range(self.n_experts):
            for j in range(self.top_k):
                sel = (topk_idx[:, j] == e)
                if not torch.any(sel):
                    continue
                x_e = x_valid[sel]                 # [Ne, D]
                y_e = self.experts[e](x_e)         # [Ne, D]
                y_valid[sel] += y_e * topk_w[sel, j].unsqueeze(-1)

        # Scatter back to [B*L, D]
        y_flat = torch.zeros_like(x_flat)           # [N, D]
        y_flat.index_copy_(0, valid_idx, y_valid)
        y = y_flat.view(B, L, D)
        return y, aux_loss


# ----------------------------
# A compact "Mamba-2 style" block
# ----------------------------

class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()
        d_inner = expand * d_model

        self.norm = RMSNorm(d_model)
        # Input projection to (gate, value)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=True)

        # Local mixing (depthwise conv) on the value stream
        self.dwconv = DepthwiseConv1d(d_inner, kernel_size=d_conv)

        # Parameterized "state" mixer (lightweight SSM surrogate)
        self.A = nn.Parameter(torch.randn(d_inner, d_state) * (0.02))
        self.B = nn.Parameter(torch.randn(d_state, d_inner) * (0.02))

        # Output projection back to model dim
        self.out_proj = nn.Linear(d_inner, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, D]
        attn_mask: optional [B, L] (1 for valid, 0 for pad). Only used to zero-out after mixing.
        """
        residual = x
        x = self.norm(x)

        # GLU gating
        x = self.in_proj(x)                    # [B, L, 2*d_inner]
        gate, val = x.chunk(2, dim=-1)         # [B, L, d_inner], [B, L, d_inner]
        gate = torch.sigmoid(gate)

        # Local depthwise conv on 'val'
        val = self.dwconv(val)                 # [B, L, d_inner]

        # Lightweight "state mixing": val @ A @ B
        state = val @ self.A                   # [B, L, d_state]
        val = state @ self.B                   # [B, L, d_inner]

        y = gate * val                         # GLU combine
        y = self.out_proj(y)
        y = self.dropout(y)

        if attn_mask is not None:
            y = y * attn_mask.unsqueeze(-1).type_as(y)

        return residual + y


class SymmetricConvBranch1D(nn.Module):
    """Symmetric (non-causal) conv-only branch to complement the Mamba path."""
    def __init__(self, d_inner: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.dw = DepthwiseConv1d(d_inner, kernel_size=kernel_size)
        self.act = nn.SiLU()
        self.pw = nn.Linear(d_inner, d_inner, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.act(y)
        y = self.pw(y)
        return self.drop(y)


class Mamba2MVBlock(nn.Module):
    """
    MambaVision-inspired mixer for 1D sequences.

    Optional: a post-mixer sparse MoE FFN residual branch.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
        # MoE settings
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner = expand * d_model

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=True)

        # Path-A: Mamba-like local + "state mixing"
        self.dwconv_a = DepthwiseConv1d(d_inner, kernel_size=d_conv)
        self.A = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)
        self.B = nn.Parameter(torch.randn(d_state, d_inner) * 0.02)

        # Path-B: symmetric conv branch (non-causal)
        self.sym_branch = SymmetricConvBranch1D(d_inner, kernel_size=max(3, 2 * d_conv + 1), dropout=dropout)

        # Reduce each path to half, then concat->fuse back to d_inner
        self.to_half_a = nn.Linear(d_inner, d_inner // 2, bias=True)
        self.to_half_b = nn.Linear(d_inner, d_inner // 2, bias=True)
        self.fuse = nn.Linear(d_inner, d_inner, bias=True)

        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Optional sparse MoE FFN (post-mixer)
        self.moe = None
        self.moe_norm = None
        self._aux_loss = None
        if use_moe:
            self.moe_norm = RMSNorm(d_model)
            self.moe = MoEFeedForward(
                d_model=d_model,
                n_experts=moe_num_experts,
                top_k=moe_top_k,
                ffn_mult=moe_ffn_mult,
                dropout=moe_dropout,
            )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        h = self.norm(x)

        h = self.in_proj(h)                    # [B,L,2*d_inner]
        gate, val = h.chunk(2, dim=-1)         # [B,L,d_inner], [B,L,d_inner]
        gate = torch.sigmoid(gate)

        # Path-A
        a = self.dwconv_a(val)
        a = (a @ self.A) @ self.B

        # Path-B
        b = self.sym_branch(val)

        # Reduce -> concat -> fuse
        a = self.to_half_a(a)
        b = self.to_half_b(b)
        y = torch.cat([a, b], dim=-1)
        y = self.fuse(y)

        # GLU + out
        y = gate * y
        y = self.out_proj(y)
        y = self.dropout(y)

        if attn_mask is not None:
            y = y * attn_mask.unsqueeze(-1).type_as(y)

        x = residual + y

        # Optional MoE FFN branch
        if self.moe is not None:
            moe_out, aux = self.moe(self.moe_norm(x), attn_mask=attn_mask)
            x = x + moe_out
            self._aux_loss = aux
        else:
            self._aux_loss = x.new_zeros(())

        return x


class AttnBlock1D(nn.Module):
    """
    Late self-attention block for 1D sequences.

    Optional: replace the dense MLP with a sparse MoE FFN.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: Optional[int] = None,
        dropout: float = 0.0,
        # MoE settings
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)

        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, self.num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)

        # Dense MLP (kept for backward compatibility / ablations)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=True),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=True),
            nn.Dropout(dropout),
        )

        # Optional sparse MoE FFN
        self.moe = None
        self._aux_loss = None
        if use_moe:
            self.moe = MoEFeedForward(
                d_model=d_model,
                n_experts=moe_num_experts,
                top_k=moe_top_k,
                ffn_mult=moe_ffn_mult,
                dropout=moe_dropout,
            )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        kpm = None
        if attn_mask is not None:
            kpm = ~(attn_mask.bool())  # True for padding
        y, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        y = self.proj(y)
        y = self.drop(y)
        x = x + y

        # FFN (dense or MoE)
        if self.moe is not None:
            moe_out, aux = self.moe(self.norm2(x), attn_mask=attn_mask)
            x = x + moe_out
            self._aux_loss = aux
        else:
            x = x + self.mlp(self.norm2(x))
            self._aux_loss = x.new_zeros(())

        return x


# ----------------------------
# Student Encoder
# ----------------------------

class StudentMamba2(nn.Module):
    """
    Student DNA encoder with Mamba-2 style blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_layers: int = 24,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
        # MoE settings
        use_moe: bool = False,
        use_moe_mamba: Optional[bool] = None,
        use_moe_attn: Optional[bool] = None,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
        moe_aux_weight: float = 1e-2,
        tie_lm_head: bool = False,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # MoE configuration (auxiliary loss is exposed in forward()).
        # Per-block MoE toggles:
        # - If both are None, follow legacy behavior: `use_moe` applies to ALL blocks.
        # - Otherwise, enable MoE only where explicitly requested.
        if (use_moe_mamba is None) and (use_moe_attn is None):
            use_moe_mamba = use_moe
            use_moe_attn = use_moe
        else:
            use_moe_mamba = bool(use_moe_mamba) if use_moe_mamba is not None else False
            use_moe_attn = bool(use_moe_attn) if use_moe_attn is not None else False

        self.use_moe_mamba = bool(use_moe_mamba)
        self.use_moe_attn = bool(use_moe_attn)
        self.use_moe = bool(self.use_moe_mamba or self.use_moe_attn)
        self.moe_aux_weight = moe_aux_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Hybrid stack: first 3/4 mixer blocks, last 1/4 attention blocks
        K = max(0, n_layers // 4)
        M = n_layers - K

        blocks = []
        for _ in range(M):
            blocks.append(Mamba2MVBlock(
                d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout,
                use_moe=self.use_moe_mamba, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
                moe_ffn_mult=moe_ffn_mult, moe_dropout=moe_dropout,
            ))
        for _ in range(K):
            blocks.append(AttnBlock1D(
                d_model=d_model, num_heads=max(1, d_model // 64), dropout=dropout,
                use_moe=self.use_moe_attn, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
                moe_ffn_mult=moe_ffn_mult, moe_dropout=moe_dropout,
            ))

        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model)

        # LM head for masked LM or distillation
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.token_emb.weight

        # Projections to teacher dims for KD (optional; created lazily)
        self.proj_to_teacher_embed = None
        self.proj_to_teacher_hidden = None

    def set_teacher_dims(self, d_teacher_embed: int, d_teacher_hidden: Optional[int] = None):
        """Create projection layers once teacher dimensions are known."""
        if d_teacher_hidden is None:
            d_teacher_hidden = d_teacher_embed
        device = next(self.parameters()).device
        self.proj_to_teacher_embed = nn.Linear(self.d_model, d_teacher_embed, bias=False).to(device)
        self.proj_to_teacher_hidden = nn.Linear(self.d_model, d_teacher_hidden, bias=False).to(device)
        nn.init.xavier_uniform_(self.proj_to_teacher_embed.weight)
        nn.init.xavier_uniform_(self.proj_to_teacher_hidden.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden: bool = False,
        compute_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        tok_emb = self.token_emb(input_ids)  # [B, L, D]
        x = tok_emb
        moe_aux = x.new_zeros(())
        all_h = []

        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask)
            if hasattr(blk, "_aux_loss") and blk._aux_loss is not None:
                moe_aux = moe_aux + blk._aux_loss
            if return_all_hidden:
                all_h.append(x)

        x = self.norm(x)
        pooled = x[:, 0] if (input_ids.size(1) > 0) else x.mean(dim=1)

        out: Dict[str, torch.Tensor] = {
            "token_embeds": tok_emb,
            "last_hidden": x,
            "pooled": pooled,
            "moe_aux_loss": moe_aux,
        }

        if compute_logits:
            out["logits"] = self.lm_head(x)

        if return_all_hidden:
            out["all_hidden"] = all_h

        return out

    # --------- KD utilities (kept to match your stage-1 code) ---------

    @staticmethod
    def _gather_positions(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Gather along sequence dimension (dim=1) at given positions."""
        assert pos.dtype in (torch.int32, torch.int64), "positions must be int tensor"
        assert t.dim() >= 3, "expected [B, L, ...]"
        B, L = t.shape[:2]
        Bp, M = pos.shape
        assert Bp == B, "batch size mismatch"

        tail_dims = t.shape[2:]
        view_shape = (B, M) + (1,) * (t.dim() - 2)
        expand_shape = (B, M) + tail_dims
        idx = pos.view(*view_shape).expand(*expand_shape).to(device=t.device)
        return t.gather(dim=1, index=idx)

    def kd_losses(
        self,
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
        masked_positions: torch.Tensor,
        tau: float = 2.0,
        lambda_embed: float = 1.0,
        lambda_hidden: float = 1.0,
        lambda_kl: float = 1.0,
        student_logits_masked: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Embedding MSE + hidden MSE + masked KL, identical to your original design."""
        assert self.proj_to_teacher_embed is not None and self.proj_to_teacher_hidden is not None, \
            "Call set_teacher_dims() before computing KD."

        # 1) Embedding KD
        s_tok = student_out["token_embeds"]
        t_tok = teacher_out["input_embeds"].detach()
        s_tok_proj = self.proj_to_teacher_embed(s_tok)
        loss_embed = F.mse_loss(s_tok_proj, t_tok)

        # 2) Encoder KD
        s_hid = student_out["last_hidden"]
        t_hid = teacher_out["last_hidden"].detach()
        s_hid_proj = self.proj_to_teacher_hidden(s_hid)
        loss_hidden = F.mse_loss(s_hid_proj, t_hid)

        # 3) Masked distribution KD
        assert teacher_out["masked_logits"] is not None, "teacher masked_logits is required"
        B, M = masked_positions.shape
        logits_t = teacher_out["masked_logits"].detach()  # [B, M, V]

        if student_logits_masked is None:
            logits_s_full = student_out["logits"]
            logits_s_masked = self._gather_positions(logits_s_full, masked_positions)
        else:
            ls = student_logits_masked
            logits_s_masked = ls if ls.size(1) == M else self._gather_positions(ls, masked_positions)

        V = logits_t.size(-1)
        logits_s_masked = logits_s_masked.reshape(B * M, V)
        logits_t_masked = logits_t.reshape(B * M, V)

        log_q = F.log_softmax(logits_s_masked / tau, dim=-1)
        log_p = F.log_softmax(logits_t_masked / tau, dim=-1)
        p = torch.exp(log_p)
        kl_vec = torch.sum(p * (log_p - log_q), dim=-1)
        loss_kl = (tau * tau) * kl_vec.mean()

        loss = lambda_embed * loss_embed + lambda_hidden * loss_hidden + lambda_kl * loss_kl

        # Optional MoE auxiliary loss term
        if self.use_moe and self.moe_aux_weight > 0:
            loss = loss + self.moe_aux_weight * student_out.get("moe_aux_loss", loss.new_zeros(()))

        return {
            "loss_total": loss,
            "loss_embed": loss_embed,
            "loss_hidden": loss_hidden,
            "loss_kl": loss_kl,
        }


# ----------------------------
# Track predictor for stage-2: sequence -> multi-track profile regression
# ----------------------------

class ConvDownsampleStem(nn.Module):
    """
    Token-level downsampling stem (Enformer-style).
    Strided Conv1d blocks achieve an exact downsample factor (e.g., 16x).
    """
    def __init__(self, d_model: int, downsample_factor: int = 16, dropout: float = 0.0):
        super().__init__()
        assert downsample_factor in {2, 4, 8, 16, 32}, "Use power-of-two downsample for simplicity."
        n_stages = int(math.log2(downsample_factor))

        blocks = []
        for _ in range(n_stages):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, stride=2, padding=2, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def downsample_mask(attn_mask: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample a binary attention mask by max-pooling within each factor window."""
        B, L = attn_mask.shape
        L2 = (L // factor) * factor
        m = attn_mask[:, :L2].view(B, L2 // factor, factor)
        return m.max(dim=-1).values

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        y = x.transpose(1, 2)  # [B, D, L]
        factor = 1
        for blk in self.blocks:
            y = blk(y)
            factor *= 2
        y = y.transpose(1, 2)  # [B, L', D]

        if attn_mask is not None:
            attn_mask = self.downsample_mask(attn_mask, factor)

        return y, attn_mask


class StudentMamba2ForTracks(nn.Module):
    """
    Enformer-like track predictor built on top of StudentMamba2-style blocks:
      - 6-mer token embedding
      - ConvDownsampleStem
      - Mamba/Attn blocks at bin-level
      - Regression head
    """
    def __init__(
        self,
        vocab_size: int,
        n_tracks: int,
        d_model: int = 384,
        n_layers: int = 24,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
        pad_id: int = 0,
        downsample_factor: int = 16,
        crop_bins_each_side: int = 0,
        output_activation: str = "none",
        # MoE settings
        use_moe: bool = False,
        use_moe_mamba: Optional[bool] = None,
        use_moe_attn: Optional[bool] = None,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
        moe_aux_weight: float = 1e-2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_tracks = n_tracks
        self.d_model = d_model
        self.pad_id = pad_id
        self.downsample_factor = downsample_factor
        self.crop_bins_each_side = crop_bins_each_side
        self.output_activation = output_activation

        # Per-block MoE toggles (same semantics as StudentMamba2).
        if (use_moe_mamba is None) and (use_moe_attn is None):
            use_moe_mamba = use_moe
            use_moe_attn = use_moe
        else:
            use_moe_mamba = bool(use_moe_mamba) if use_moe_mamba is not None else False
            use_moe_attn = bool(use_moe_attn) if use_moe_attn is not None else False

        self.use_moe_mamba = bool(use_moe_mamba)
        self.use_moe_attn = bool(use_moe_attn)
        self.use_moe = bool(self.use_moe_mamba or self.use_moe_attn)
        self.moe_aux_weight = moe_aux_weight

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.stem = ConvDownsampleStem(d_model, downsample_factor=downsample_factor, dropout=dropout)

        K = max(0, n_layers // 4)
        M = n_layers - K
        blocks = []
        for _ in range(M):
            blocks.append(Mamba2MVBlock(
                d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout,
                use_moe=self.use_moe_mamba, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
                moe_ffn_mult=moe_ffn_mult, moe_dropout=moe_dropout,
            ))
        for _ in range(K):
            blocks.append(AttnBlock1D(
                d_model=d_model, num_heads=max(1, d_model // 64), dropout=dropout,
                use_moe=self.use_moe_attn, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
                moe_ffn_mult=moe_ffn_mult, moe_dropout=moe_dropout,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model)

        self.head = nn.Linear(d_model, n_tracks, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_bins: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        x = self.token_emb(input_ids)         # [B, L, D]
        x, m = self.stem(x, attention_mask)   # [B, L', D], [B, L']

        moe_aux = x.new_zeros(())
        for blk in self.blocks:
            x = blk(x, attn_mask=m)
            if hasattr(blk, "_aux_loss") and blk._aux_loss is not None:
                moe_aux = moe_aux + blk._aux_loss
        x = self.norm(x)

        # Optional cropping
        if self.crop_bins_each_side > 0:
            c = self.crop_bins_each_side
            x = x[:, c:-c, :]
            if m is not None:
                m = m[:, c:-c]

        logits = self.head(x)
        if self.output_activation == "softplus":
            pred = F.softplus(logits) + 1e-4
        elif self.output_activation == "none":
            pred = logits
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")

        out: Dict[str, torch.Tensor] = {"pred": pred, "logits": logits, "moe_aux_loss": moe_aux}
        if self.output_activation == "softplus":
            out["rates"] = pred
        if m is not None:
            out["attn_mask_bins"] = m
        if return_bins:
            out["bin_features"] = x
        return out


class StudentMamba2ForTracksAndMLM(nn.Module):
    """
    Joint objective model for stage2:
      - MLM on token-level backbone (StudentMamba2)
      - Profile regression head on downsampled token hidden states

    This matches the 'dense pretrain -> add MoE -> continue training' idea:
      - Load stage1 dense MLM checkpoint into backbone (strict=False)
      - MoE modules are newly introduced and initialized near-zero (see MoEFeedForward)
    """
    def __init__(
        self,
        vocab_size: int,
        n_tracks: int,
        d_model: int = 384,
        n_layers: int = 24,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.0,
        pad_id: int = 0,
        downsample_factor: int = 16,
        crop_bins_each_side: int = 0,
        output_activation: str = "none",
        # MoE configs
        use_moe: bool = False,
        use_moe_mamba: Optional[bool] = None,
        use_moe_attn: Optional[bool] = None,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
        tie_lm_head: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.downsample_factor = int(downsample_factor)
        self.crop_bins_each_side = int(crop_bins_each_side)
        self.output_activation = output_activation

        # Token-level backbone for MLM
        self.backbone = StudentMamba2(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
            pad_id=pad_id,
            use_moe=use_moe,
            use_moe_mamba=use_moe_mamba,
            use_moe_attn=use_moe_attn,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_ffn_mult=moe_ffn_mult,
            moe_dropout=moe_dropout,
            tie_lm_head=tie_lm_head,
        )

        # Light bin-level head (no extra bin-level blocks, to keep it simple and stable)
        self.bin_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, n_tracks, bias=True)

    def _downsample(self, x: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Downsample token features to bins by mean pooling over contiguous groups.
        x:    [B, L, D]
        mask: [B, L] (1=valid token, 0=pad)
        returns:
          xb:  [B, T, D]
          mb:  [B, T]
        """
        B, L, D = x.shape
        s = self.downsample_factor
        if s <= 1:
            return x, mask

        T = L // s
        L_trunc = T * s
        if L_trunc < L:
            x = x[:, :L_trunc, :]
            mask = mask[:, :L_trunc]

        xb = x.reshape(B, T, s, D).mean(dim=2)
        mb = mask.reshape(B, T, s).min(dim=2).values  # bin valid only if all tokens are valid

        # Optional cropping (to match Enformer-style center-crop)
        if self.crop_bins_each_side > 0 and T > 2 * self.crop_bins_each_side:
            c = self.crop_bins_each_side
            xb = xb[:, c:-c, :]
            mb = mb[:, c:-c]
        return xb, mb

    def _apply_activation(self, pred: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "softplus":
            return F.softplus(pred)
        if self.output_activation == "relu":
            return F.relu(pred)
        return pred

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
        # --- New flags for memory optimization ---
        compute_pred: bool = True,
        logits_on_masked_only: bool = False,
        mlm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Memory-optimized forward:
          - Profile regression can be disabled via compute_pred=False.
          - MLM logits can be computed only on masked positions to avoid allocating [B, L, V].
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        # If we compute logits only on masked positions, we do NOT ask backbone to build full logits.
        backbone_compute_logits = compute_logits and (not logits_on_masked_only)

        bb = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_all_hidden=False,
            compute_logits=backbone_compute_logits,
        )
        x = bb["last_hidden"]  # [B, L, D]
        moe_aux = bb.get("moe_aux_loss", x.new_zeros(()))

        out: Dict[str, torch.Tensor] = {"moe_aux_loss": moe_aux}

        # ---- Profile head (optional) ----
        if compute_pred:
            xb, mb = self._downsample(x, attention_mask)
            pred = self.head(self.bin_norm(xb))  # [B, T, C]
            pred = self._apply_activation(pred)
            out["pred"] = pred
            out["bin_mask"] = mb

        # ---- MLM logits ----
        if logits_on_masked_only:
            if mlm_labels is None:
                raise ValueError("mlm_labels must be provided when logits_on_masked_only=True")

            # Compute logits only for masked tokens to save memory.
            # mask: [B, L] boolean
            mask = (mlm_labels != -100)
            if mask.any():
                x_masked = x[mask]  # [N_mask, D]
                logits_masked = self.backbone.lm_head(x_masked)  # [N_mask, V]
                labels_masked = mlm_labels[mask]  # [N_mask]
            else:
                # No masked tokens (rare), return empty tensors.
                logits_masked = x.new_zeros((0, self.backbone.lm_head.out_features))
                labels_masked = mlm_labels.new_zeros((0,))

            out["logits_masked"] = logits_masked
            out["labels_masked"] = labels_masked

        else:
            if compute_logits:
                logits = bb.get("logits", None)
                if logits is not None:
                    out["logits"] = logits

        return out


def poisson_nll_loss(rates: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Poisson negative log-likelihood for binned tracks."""
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="none")
    loss = loss_fn(rates, targets)  # [B, T, C]
    if mask is not None:
        loss = loss * mask.unsqueeze(-1).type_as(loss)
        denom = (mask.sum() * targets.size(-1)).clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()