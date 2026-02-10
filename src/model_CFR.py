from typing import Optional, Dict, Any
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
# A compact "Mamba-2 style" block
# NOTE:
#   This is an efficient, self-contained approximation inspired by Mamba-2:
#   - Pre-norm
#   - GLU gating
#   - Depthwise conv for local mixing
#   - Parameterized state-space mixing (simplified)
#   - Residual connection
# It does not rely on external libraries and is stable for 2k+ tokens.
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
        # We use two learned matrices to emulate scan-like mixing.
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

        # Lightweight "state mixing": val @ A @ B  (emulates selective scan)
        # Shapes: val:[B,L,d_inner], A:[d_inner,d_state], B:[d_state,d_inner]
        # Result: [B,L,d_inner]
        state = val @ self.A                   # [B, L, d_state]
        val = state @ self.B                   # [B, L, d_inner]

        y = gate * val                         # GLU combine
        y = self.out_proj(y)
        y = self.dropout(y)

        if attn_mask is not None:
            y = y * attn_mask.unsqueeze(-1).type_as(y)

        return residual + y

class SymmetricConvBranch1D(nn.Module):
    """
    Symmetric (non-causal) conv-only branch to complement the Mamba path.
    Operates at the inner width (d_inner). Depthwise + pointwise conv.
    """
    def __init__(self, d_inner: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        # reuse DepthwiseConv1d but at d_inner width
        self.dw = DepthwiseConv1d(d_inner, kernel_size=kernel_size)
        self.act = nn.SiLU()
        self.pw = nn.Linear(d_inner, d_inner, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_inner]
        y = self.dw(x)              # depthwise
        y = self.act(y)
        y = self.pw(y)              # pointwise (linear over channel)
        return self.drop(y)


class Mamba2MVBlock(nn.Module):
    """
    MambaVision-inspired mixer for 1D sequences, with the SAME public signature as Mamba2Block:
      - Pre-norm -> parallel {Mamba2-like path, SymmetricConv path}
      - Each path -> project to d_inner//2; concat -> fuse (back to d_inner)
      - GLU gating (same gate as student original) -> out_proj -> dropout
      - Residual add; returns [B,L,D]
    NOTE: No extra FFN/residual stage is added, to keep the student block "shape" similar.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, d_conv: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner = expand * d_model

        self.norm = RMSNorm(d_model)
        # project to (gate, val) same as original student block
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=True)

        # --- Path-A: Mamba2-like local + "state mixing" (kept from original) ---
        self.dwconv_a = DepthwiseConv1d(d_inner, kernel_size=d_conv)
        self.A = nn.Parameter(torch.randn(d_inner, d_state) * 0.02)
        self.B = nn.Parameter(torch.randn(d_state, d_inner) * 0.02)

        # --- Path-B: symmetric conv branch (non-causal) ---
        # Use a slightly larger kernel for the symmetric path
        self.sym_branch = SymmetricConvBranch1D(d_inner, kernel_size=max(3, 2 * d_conv + 1), dropout=dropout)

        # reduce each path to half, then concat->fuse back to d_inner
        self.to_half_a = nn.Linear(d_inner, d_inner // 2, bias=True)
        self.to_half_b = nn.Linear(d_inner, d_inner // 2, bias=True)
        self.fuse = nn.Linear(d_inner, d_inner, bias=True)

        # final projection to model dim (kept) + dropout
        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, L, D]; attn_mask: [B, L] with 1 for valid (kept for API symmetry)
        """
        residual = x
        h = self.norm(x)

        # input projection and GLU split (shared gate)
        h = self.in_proj(h)                    # [B,L,2*d_inner]
        gate, val = h.chunk(2, dim=-1)         # [B,L,d_inner], [B,L,d_inner]
        gate = torch.sigmoid(gate)

        # Path-A: Mamba-like
        a = self.dwconv_a(val)                 # [B,L,d_inner]
        a = (a @ self.A) @ self.B              # state mixing surrogate -> [B,L,d_inner]

        # Path-B: symmetric conv (non-causal)
        b = self.sym_branch(val)               # [B,L,d_inner]

        # reduce -> concat -> fuse
        a = self.to_half_a(a)                  # [B,L,d_inner//2]
        b = self.to_half_b(b)                  # [B,L,d_inner//2]
        y = torch.cat([a, b], dim=-1)          # [B,L,d_inner]
        y = self.fuse(y)                       # [B,L,d_inner]

        # GLU and out
        y = gate * y
        y = self.out_proj(y)
        y = self.dropout(y)

        if attn_mask is not None:
            y = y * attn_mask.unsqueeze(-1).type_as(y)

        return residual + y

class AttnBlock1D(nn.Module):
    """
    Late self-attention block for 1D sequences.
    Signature matches Mamba2Block/Mamba2MVBlock: forward(x, attn_mask=None) -> [B,L,D]
    """
    def __init__(self, d_model: int, num_heads: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads or max(1, d_model // 64)

        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, self.num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

        # keep a light MLP but not adding an extra API; stays internal to the block
        self.norm2 = RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=True),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,L,D]; attn_mask: [B,L] with 1 for valid (we convert to key_padding_mask)
        """
        # pre-norm
        h = self.norm1(x)
        # key_padding_mask: True means to mask (i.e., invalid positions)
        kpm = None
        if attn_mask is not None:
            kpm = ~(attn_mask.bool())  # [B,L], True for padding
        y, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        y = self.proj(y)
        y = self.drop(y)
        x = x + y

        # light FFN
        x = x + self.mlp(self.norm2(x))
        return x

# ----------------------------
# Student Encoder (Mamba-2 style)
# ----------------------------
class StudentMamba2(nn.Module):
    """
    Student DNA encoder with Mamba-2 style blocks.
    It must share the tokenizer with NT-500M (vocab_size and special tokens).
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
        tie_lm_head: bool = False,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # Token embedding shared with tokenizer ids
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # # Stack of Mamba-2 blocks
        # self.blocks = nn.ModuleList([
        #     Mamba2Block(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout)
        #     for _ in range(n_layers)
        # ])

        # Hybrid stack:
        #   first N-K layers: Mamba2MVBlock (Mamba + symmetric conv, concat-fuse)
        #   last  K layers:   AttnBlock1D (late attention for long-range aggregation)
        K = max(0, n_layers // 4)  # no new hyperparam; default late-attn ratio = 1/4
        M = n_layers - K

        blocks = []
        for _ in range(M):
            blocks.append(Mamba2MVBlock(d_model=d_model, d_state=d_state, expand=expand,
                                        d_conv=d_conv, dropout=dropout))
        for _ in range(K):
            blocks.append(AttnBlock1D(d_model=d_model, num_heads=max(1, d_model // 64),
                                      dropout=dropout))

        self.blocks = nn.ModuleList(blocks)

        self.norm = RMSNorm(d_model)

        # LM head for masked distribution KD
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.token_emb.weight  # optional weight tying

        # Projections to teacher dims for KD (filled lazily when teacher dims are known)
        self.proj_to_teacher_embed = None  # nn.Linear(d_model, d_teacher_embed)
        self.proj_to_teacher_hidden = None # nn.Linear(d_model, d_teacher_hidden)

    # --------- Utilities ---------

    def set_teacher_dims(self, d_teacher_embed: int, d_teacher_hidden: Optional[int] = None):
        """Create projection layers once teacher dimensions are known, on the same device as the student."""
        if d_teacher_hidden is None:
            d_teacher_hidden = d_teacher_embed
        device = next(self.parameters()).device  # student's current device (cpu/cuda)

        self.proj_to_teacher_embed = nn.Linear(self.d_model, d_teacher_embed, bias=False).to(device)
        self.proj_to_teacher_hidden = nn.Linear(self.d_model, d_teacher_hidden, bias=False).to(device)

        # (optional) good initialization
        nn.init.xavier_uniform_(self.proj_to_teacher_embed.weight)
        nn.init.xavier_uniform_(self.proj_to_teacher_hidden.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_emb

    # --------- Forward ---------
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_all_hidden: bool = False,
            compute_logits: bool = True,  # NEW: skip LM head when False to save memory
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: [B, L]
        attention_mask: [B, L] with 1 for valid, 0 for padding
        return_all_hidden: if True, also return intermediate layer outputs
        compute_logits: if False, do NOT build [B, L, V] logits (saves lots of memory)
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()  # explicit pad id

        # Token embeddings
        tok_emb = self.token_emb(input_ids)  # [B, L, D]

        # Mamba-2 stack
        x = tok_emb
        all_h = []
        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask)
            if return_all_hidden:
                all_h.append(x)

        x = self.norm(x)  # [B, L, D]
        pooled = x[:, 0] if (input_ids.size(1) > 0) else x.mean(dim=1)

        out = {
            "token_embeds": tok_emb,  # [B, L, D]
            "last_hidden": x,  # [B, L, D]
            "pooled": pooled,  # [B, D]
        }

        # Only compute logits when explicitly requested
        if compute_logits:
            out["logits"] = self.lm_head(x)  # [B, L, V]

        if return_all_hidden:
            out["all_hidden"] = all_h

        return out

    # --------- KD Losses ---------
    @staticmethod
    def _gather_positions(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Gather along sequence dimension (dim=1) at given positions.
        t:   [B, L, *S]   (e.g., logits with *S = [V])
        pos: [B, M]       integer positions in [0, L)
        return: [B, M, *S]
        """
        assert pos.dtype in (torch.int32, torch.int64), "positions must be int tensor"
        assert t.dim() >= 3, "expected [B, L, ...]"
        B, L = t.shape[:2]
        Bp, M = pos.shape
        assert Bp == B, "batch size mismatch between t and pos"

        # Build index of shape [B, M, *S] by expanding over the tail dims of t
        # Start from [B, M, 1, 1, ...] and expand to [B, M, *S]
        tail_dims = t.shape[2:]  # *S
        view_shape = (B, M) + (1,) * (t.dim() - 2)
        expand_shape = (B, M) + tail_dims

        idx = pos.view(*view_shape).expand(*expand_shape).to(device=t.device)

        # Gather along dim=1 using the expanded index
        out = t.gather(dim=1, index=idx)
        return out

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
        """
        Compute three KD losses:
          - Embedding KD (token embeddings): MSE(student_emb -> teacher_emb)
          - Encoder KD (last hidden states): MSE(student_hidden -> teacher_hidden)
          - Masked distribution KD: KL(student_logits_masked || teacher_logits_masked), temperature tau
        Requirements:
          - self.proj_to_teacher_embed / hidden must be initialized by set_teacher_dims().
          - student_out: from self.forward()
          - teacher_out: dict from TeacherNT500M.forward()
          - masked_positions: [B, M] with valid indices (teacher/student use the same tokenizer)
        """
        assert self.proj_to_teacher_embed is not None and self.proj_to_teacher_hidden is not None, \
            "Call set_teacher_dims() with teacher dims before computing KD."

        # 1) Embedding KD
        s_tok = student_out["token_embeds"]                  # [B, L, Ds]
        t_tok = teacher_out["input_embeds"].detach()         # [B, L, Dt]
        s_tok_proj = self.proj_to_teacher_embed(s_tok)       # [B, L, Dt]
        loss_embed = F.mse_loss(s_tok_proj, t_tok)

        # 2) Encoder KD
        s_hid = student_out["last_hidden"]                   # [B, L, Ds]
        t_hid = teacher_out["last_hidden"].detach()          # [B, L, Dt]
        s_hid_proj = self.proj_to_teacher_hidden(s_hid)      # [B, L, Dt]
        loss_hidden = F.mse_loss(s_hid_proj, t_hid)

        # 3) Masked distribution KD
        #    Gather student/teacher logits at the same masked positions.
        #    Shapes: logits_s_masked/logits_t_masked -> [B*M, V]
        assert teacher_out["masked_logits"] is not None, "teacher masked_logits is required for KL KD."
        B, M = masked_positions.shape
        logits_t = teacher_out["masked_logits"].detach()  # [B, M, V]
        assert logits_t.dim() == 3 and logits_t.size(1) == M, "teacher masked_logits must be [B,M,V]"

        # Student masked logits handling
        if student_logits_masked is None:
            logits_s_full = student_out["logits"]  # [B, L, V]
            logits_s_masked = self._gather_positions(logits_s_full, masked_positions)  # [B, M, V]
        else:
            ls = student_logits_masked  # [B, L, V] or [B, M, V]
            logits_s_masked = ls if ls.size(1) == M else self._gather_positions(ls, masked_positions)
        # then reshape to [B*M, V] as you do:
        V = logits_t.size(-1)
        logits_s_masked = logits_s_masked.reshape(B * M, V)

        logits_t_masked = logits_t.reshape(B * M, V)

        # Temperature-scaled KL: KL(p_teacher || q_student) >= 0
        log_q = F.log_softmax(logits_s_masked / tau, dim=-1)  # log q_s
        log_p = F.log_softmax(logits_t_masked / tau, dim=-1)  # log p_t
        p = torch.exp(log_p)  # p_t
        kl_vec = torch.sum(p * (log_p - log_q), dim=-1)  # [B*M]
        loss_kl = (tau * tau) * kl_vec.mean()
        # Weighted sum
        loss = lambda_embed * loss_embed + lambda_hidden * loss_hidden + lambda_kl * loss_kl

        return {
            "loss_total": loss,
            "loss_embed": loss_embed,
            "loss_hidden": loss_hidden,
            "loss_kl": loss_kl
        }

import math
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDownsampleStem(nn.Module):
    """
    Token-level downsampling stem.
    This mimics Enformer-style convolution/pooling to reduce sequence length
    before long-range mixing.

    We use strided Conv1d blocks to achieve an exact downsample factor (e.g., 16x).
    """
    def __init__(self, d_model: int, downsample_factor: int = 16, dropout: float = 0.0):
        super().__init__()
        assert downsample_factor in {2,4,8,16,32}, "Use power-of-two downsample for simplicity."
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
        """
        Downsample a binary attention mask by max-pooling within each factor window.
        attn_mask: [B, L] with 1 for valid, 0 for pad.
        return: [B, L//factor]
        """
        B, L = attn_mask.shape
        L2 = (L // factor) * factor
        m = attn_mask[:, :L2].view(B, L2 // factor, factor)
        m = m.max(dim=-1).values
        return m

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        x: [B, L, D]
        attn_mask: [B, L] (optional)
        """
        y = x.transpose(1, 2)  # [B, D, L]
        factor = 1
        for blk in self.blocks:
            y = blk(y)          # stride-2 conv
            factor *= 2
        y = y.transpose(1, 2)  # [B, L', D]

        if attn_mask is not None:
            attn_mask = self.downsample_mask(attn_mask, factor)

        return y, attn_mask


class StudentMamba2ForTracks(nn.Module):
    """
    Enformer-like track predictor built on top of the existing StudentMamba2 blocks:
      - 6-mer token embedding
      - ConvDownsampleStem (e.g., 16x -> 96bp bins if token=6bp)
      - Mamba/Attn blocks at bin-level
      - Cropping (optional)
      - Track head: regression head (optionally with non-negative activation)
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
        downsample_factor: int = 16,  # 16 tokens/bin -> 96bp bins for non-overlap 6-mer
        crop_bins_each_side: int = 0, # set >0 to mimic Enformer cropping
        output_activation: str = "none", # "none" for regression; "softplus" for non-negative outputs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_tracks = n_tracks
        self.d_model = d_model
        self.pad_id = pad_id
        self.downsample_factor = downsample_factor
        self.crop_bins_each_side = crop_bins_each_side
        self.output_activation = output_activation

        # Token embedding (same as student)
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Downsampling stem
        self.stem = ConvDownsampleStem(d_model, downsample_factor=downsample_factor, dropout=dropout)

        # Reuse the same block definitions from student.py
        K = max(0, n_layers // 4)
        M = n_layers - K
        blocks = []
        for _ in range(M):
            blocks.append(Mamba2MVBlock(d_model=d_model, d_state=d_state, expand=expand, d_conv=d_conv, dropout=dropout))
        for _ in range(K):
            blocks.append(AttnBlock1D(d_model=d_model, num_heads=max(1, d_model // 64), dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model)

        # Track head: regression outputs (activation controlled by output_activation)
        self.head = nn.Linear(d_model, n_tracks, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_bins: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: [B, L_tokens]
        attention_mask: [B, L_tokens] (1 valid, 0 pad)
        Returns:
          pred: [B, T_bins, n_tracks] (regression outputs)
          attn_mask_bins: [B, T_bins]
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).long()

        x = self.token_emb(input_ids)  # [B, L, D]
        x, m = self.stem(x, attention_mask)  # [B, L/df, D], [B, L/df]

        for blk in self.blocks:
            x = blk(x, attn_mask=m)
        x = self.norm(x)

        # Optional cropping to reduce boundary effects
        if self.crop_bins_each_side > 0:
            c = self.crop_bins_each_side
            x = x[:, c:-c, :]
            if m is not None:
                m = m[:, c:-c]
        # Regression head (optionally enforce non-negativity)
        logits = self.head(x)  # [B, T, n_tracks]
        if self.output_activation == "softplus":
            pred = F.softplus(logits) + 1e-4  # strictly positive, if you still want non-negative regression
        elif self.output_activation == "none":
            pred = logits
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")

        out = {"pred": pred, "logits": logits}
        # Backward-compatibility: expose 'rates' when using softplus
        if self.output_activation == "softplus":
            out["rates"] = pred
        if m is not None:
            out["attn_mask_bins"] = m
        if return_bins:
            out["bin_features"] = x
        return out


def poisson_nll_loss(rates: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Poisson negative log-likelihood for binned tracks.
    rates:   [B, T, C] positive
    targets: [B, T, C] non-negative
    mask:    [B, T] optional (1 valid, 0 pad)
    """
    # torch PoissonNLLLoss expects input as rate when log_input=False
    loss_fn = nn.PoissonNLLLoss(log_input=False, reduction="none")
    loss = loss_fn(rates, targets)  # [B, T, C]

    if mask is not None:
        loss = loss * mask.unsqueeze(-1).type_as(loss)

        denom = mask.sum() * targets.size(-1)
        denom = denom.clamp_min(1.0)
        return loss.sum() / denom

    return loss.mean()
