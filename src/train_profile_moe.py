# train_profile_moe_joint_checked.py
# Stage-2 joint supervised pretraining for sequence -> multi-track profile regression + MLM.
# This script follows the "dense pretrain -> MoEfication -> continued training" idea:
#   - Load a dense Stage-1 MLM checkpoint into the backbone (strict=False).
#   - Enable MoE ONLY inside late Attention blocks in Stage-2 (keep Mamba blocks dense).
#   - Train with a joint objective: profile regression + MLM (+ MoE load-balancing aux loss).
#
# English comments only.

import os
import math
import argparse
import random
import time
from glob import glob
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import contextlib
from contextlib import nullcontext
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from model_FRE import StudentMamba2ForTracksAndMLM, StudentMamba2  # MoE-enabled models (attn-only MoE)


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove a leading 'module.' prefix (common when saving from DDP)."""
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _extract_state(ckpt_obj: object) -> dict:
    """Extract a PyTorch state_dict from a variety of checkpoint formats."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "module", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")


@torch.no_grad()
def upcycle_dense_attn_mlp_to_moe(backbone: torch.nn.Module, verbose: bool = False):
    """
    Sparse upcycling:
    copy the dense AttnBlock MLP weights into every expert of the MoE FFN.

    Notes:
      - This is function-preserving when router logits are near-uniform and experts are identical.
      - We only upcycle blocks that have BOTH `mlp` (dense FFN) and `moe` (sparse FFN).
    """
    n_copied = 0
    for blk in getattr(backbone, "blocks", []):
        moe = getattr(blk, "moe", None)
        mlp = getattr(blk, "mlp", None)
        if moe is None or mlp is None:
            continue

        # Expect dense mlp = [Linear, SiLU, Linear, Dropout]
        if not hasattr(moe, "experts") or len(moe.experts) == 0:
            continue
        if not (isinstance(mlp, torch.nn.Sequential) and len(mlp) >= 3):
            continue

        fc1 = mlp[0]
        fc2 = mlp[2]
        if not (isinstance(fc1, torch.nn.Linear) and isinstance(fc2, torch.nn.Linear)):
            continue

        for exp in moe.experts:
            exp.fc1.weight.copy_(fc1.weight)
            if exp.fc1.bias is not None and fc1.bias is not None:
                exp.fc1.bias.copy_(fc1.bias)
            exp.fc2.weight.copy_(fc2.weight)
            if exp.fc2.bias is not None and fc2.bias is not None:
                exp.fc2.bias.copy_(fc2.bias)

        # Make router logits uniform at init (ties -> stable top-k selection).
        if hasattr(moe, "router") and hasattr(moe.router, "proj"):
            torch.nn.init.zeros_(moe.router.proj.weight)

        n_copied += 1

    if verbose:
        print(f"[upcycle] copied dense MLP -> MoE experts for {n_copied} attention blocks")
    return n_copied


def setup_distributed_if_needed():
    """Initialize torch.distributed from torchrun env vars if WORLD_SIZE > 1."""
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


class NPZProfileDataset(Dataset):
    """
    Load pre-built NPZ shards:
      - seq: uint8 array of ASCII-encoded DNA string (fixed length in bp)
      - y:   float32 array [T_bins, C] (already binned & transformed)
    """
    def __init__(self, npz_dir: str):
        self.files = sorted(glob(os.path.join(npz_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz found under {npz_dir}")

        # Build an index mapping global idx -> (file_idx, row_idx)
        self.index = []
        self._cache = None
        self._cache_i = -1

        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = z["y"].shape[0]
            for ri in range(n):
                self.index.append((fi, ri))

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi: int):
        if self._cache_i != fi:
            if self._cache is not None:
                # Close previous NPZ handle to avoid file descriptor growth.
                try:
                    self._cache.close()
                except Exception:
                    pass
            self._cache = np.load(self.files[fi])
            self._cache_i = fi

    def __getitem__(self, idx: int):
        fi, ri = self.index[idx]
        self._load_file(fi)
        seq_bytes = self._cache["seq"][ri]     # [bp] uint8, ASCII codes
        y = self._cache["y"][ri]               # [T, C] float32
        seq = bytes(seq_bytes.tolist()).decode("ascii")
        return seq, y


def build_collate_fn(tokenizer, max_tokens: int):
    """Tokenize sequences to fixed length and stack targets."""
    def _collate(batch):
        seqs, ys = zip(*batch)
        enc = tokenizer(
            list(seqs),
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        )
        y = torch.tensor(np.stack(ys, axis=0), dtype=torch.float32)  # [B,T,C]
        return {
            "input_ids": enc["input_ids"].long(),
            "attention_mask": enc["attention_mask"].long(),
            "targets": y,
        }
    return _collate


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    pred/target: [B,T,C]
    mask: [B,T] with 1 for valid bins (optional)
    """
    loss = F.smooth_l1_loss(pred, target, reduction="none")  # [B,T,C]
    if mask is None:
        return loss.mean()
    loss = loss * mask.unsqueeze(-1).type_as(loss)
    denom = (mask.sum() * target.size(-1)).clamp_min(1.0)
    return loss.sum() / denom


def make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_prob: float = 0.15,
    mask_prob: float = 0.8,
    random_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masked inputs and MLM labels following BERT-style masking.
    Returns:
      masked_input_ids: [B, L]
      labels:          [B, L] with -100 for non-MLM positions
    """
    device = input_ids.device
    labels = input_ids.clone()

    # Do not mask padding: attention_mask is 1 for tokens, 0 for padding.
    probability_matrix = torch.full(labels.shape, mlm_prob, device=device) * attention_mask.float()
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # ignore

    masked_input_ids = input_ids.clone()

    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_prob, device=device)).bool() & masked_indices
    masked_input_ids[indices_replaced] = mask_token_id

    # 10% replace with random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, random_prob, device=device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_tokens = torch.randint(low=0, high=vocab_size, size=labels.shape, device=device, dtype=torch.long)
    masked_input_ids[indices_random] = random_tokens[indices_random]

    # Remaining 10% keep original token
    return masked_input_ids, labels


def main():
    ap = argparse.ArgumentParser()

    # NOTE: --npz_dir / --n_tracks are required for training, but optional for --upcycle_only.
    ap.add_argument("--npz_dir", type=str, default=None, help="Directory of NPZ shards for (seq, binned targets).")
    ap.add_argument("--teacher_dir", type=str, required=True, help="Tokenizer source (NT-v2 dir).")
    ap.add_argument("--resume_path", type=str, required=True, help="Stage1 MLM checkpoint (dense or MoE student).")
    ap.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output checkpoint path for training. Not required when --upcycle_only is set.",
    )

    # Dense -> MoE upcycling
    ap.add_argument(
        "--upcycle_ckpt_out",
        type=str,
        default=None,
        help=(
            "If set, save an upcycled MoE backbone checkpoint right after loading resume_path. "
            "This is useful to create a MoE stage-1 checkpoint from a dense stage-1 checkpoint."
        ),
    )
    ap.add_argument(
        "--upcycle_only",
        action="store_true",
        help="Only perform dense->MoE upcycling (and optionally save), then exit without training.",
    )
    ap.add_argument(
        "--skip_upcycle_mlp_copy",
        action="store_true",
        help=(
            "If set, do NOT copy dense attention MLP weights into MoE experts (use default MoE init instead). "
            "By default, when --use_moe is enabled and the resume checkpoint is dense, we upcycle the attention MLP."
        ),
    )

    ap.add_argument("--seq_len", type=int, default=2048, help="Token length (NT non-overlap 6-mer => bp=seq_len*6).")
    ap.add_argument("--n_tracks", type=int, default=None)

    # Regression settings
    ap.add_argument("--loss", type=str, default="smoothl1", choices=["smoothl1", "mse"],
                    help="Regression loss for profile supervision.")
    ap.add_argument("--output_activation", type=str, default="none", choices=["none", "softplus"],
                    help="Output activation in the track head. Use 'none' for pure regression.")

    # Backbone settings
    ap.add_argument("--d_model", type=int, default=448)
    ap.add_argument("--n_layers", type=int, default=28)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    # MoE settings
    ap.add_argument("--use_moe", action="store_true", help="Enable sparse MoE FFN inside the encoder.")
    ap.add_argument("--moe_num_experts", type=int, default=8)
    ap.add_argument("--moe_top_k", type=int, default=2, choices=[1, 2])
    ap.add_argument("--moe_ffn_mult", type=float, default=4.0)
    ap.add_argument("--moe_dropout", type=float, default=0.0)
    ap.add_argument("--moe_aux_weight", type=float, default=1e-2,
                    help="Weight for MoE load-balancing auxiliary loss (added to total loss).")

    # Joint MLM + profile objective (stage2)
    ap.add_argument("--mlm_weight", type=float, default=0.2, help="Weight for MLM loss in stage2.")
    ap.add_argument("--mlm_prob", type=float, default=0.15, help="Masking probability for MLM.")
    ap.add_argument("--mlm_mask_prob", type=float, default=0.8, help="P([MASK] | masked).")
    ap.add_argument("--mlm_random_prob", type=float, default=0.1, help="P(random | masked).")

    # Binning / cropping (must match your target NPZ preprocessing)
    ap.add_argument("--downsample_factor", type=int, default=16, help="Tokens per bin.")
    ap.add_argument("--crop_bins_each_side", type=int, default=0)

    # Optimization
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # AMP / devices
    ap.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--gpu_ids", type=str, default=None, help="Optional CUDA_VISIBLE_DEVICES override (e.g., '0,1,2').")

    # TensorBoard
    ap.add_argument("--tb_logdir", type=str, default="runs/profile_moe", help="TensorBoard log root dir.")
    ap.add_argument(
        "--tb_run_name",
        type=str,
        default=None,
        help=(
            "Optional TensorBoard run name (subdir under --tb_logdir). "
            "If not set, use '<save_stem>_<timestamp>'."
        ),
    )
    ap.add_argument("--tb_log_every", type=int, default=10, help="Log scalars every N optimizer steps.")
    ap.add_argument("--tb_flush_secs", type=int, default=30, help="SummaryWriter flush_secs.")
    ap.add_argument("--tb_disable", action="store_true", help="Disable TensorBoard logging.")

    args = ap.parse_args()

    # MoE placement policy for stage-2:
    # Enable MoE ONLY in late Attention blocks; keep Mamba mixer blocks dense.
    use_moe_attn = bool(args.use_moe)
    use_moe_mamba = False


    grad_clip = float(getattr(args, "grad_clip", getattr(args, "clip_grad", 0.0)))

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Reproducibility (best-effort; distributed + cudnn can still introduce nondeterminism)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    is_dist, rank, world, local_rank = setup_distributed_if_needed()
    device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_dir, use_fast=True, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must have pad_token_id.")

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask_token_id; MLM requires a [MASK] token.")

    vocab_size = tokenizer.vocab_size

    # ---- Upcycle-only mode: convert a dense stage-1 checkpoint to a MoE stage-1 checkpoint, then exit. ----
    if args.upcycle_only:
        if not args.use_moe:
            raise ValueError("--upcycle_only requires --use_moe (otherwise there is no MoE to create).")

        # Default output path if not provided.
        if args.upcycle_ckpt_out is None:
            stem, ext = os.path.splitext(args.resume_path)
            args.upcycle_ckpt_out = f"{stem}_moeE{args.moe_num_experts}{ext or '.pt'}"

        bb = StudentMamba2(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            expand=args.expand,
            d_conv=args.d_conv,
            dropout=args.dropout,
            pad_id=pad_id,
            use_moe=True,
            use_moe_mamba=use_moe_mamba,
            use_moe_attn=use_moe_attn,
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            moe_ffn_mult=args.moe_ffn_mult,
            moe_dropout=args.moe_dropout,
        ).to(device)

        ckpt_obj = torch.load(args.resume_path, map_location="cpu")
        state = _strip_module_prefix(_extract_state(ckpt_obj))
        has_moe_params = any(".moe." in k or ".router." in k for k in state.keys())
        missing, unexpected = bb.load_state_dict(state, strict=False)

        if args.use_moe and (not has_moe_params) and (not args.skip_upcycle_mlp_copy):
            upcycle_dense_attn_mlp_to_moe(bb, verbose=is_main(rank))

        if is_main(rank):
            torch.save(
                {
                    "model": bb.state_dict(),
                    "config": vars(args),
                    "note": "dense->moe upcycled stage-1 backbone checkpoint",
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                },
                args.upcycle_ckpt_out,
            )
            print(f"[upcycle_only] saved -> {args.upcycle_ckpt_out} (detected_moe_params={has_moe_params})")

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    # Training mode sanity checks.
    if args.npz_dir is None:
        raise ValueError("--npz_dir is required for training (not in --upcycle_only mode).")
    if args.n_tracks is None:
        raise ValueError("--n_tracks is required for training (not in --upcycle_only mode).")
    if args.save_path is None:
        raise ValueError("--save_path is required for training (not in --upcycle_only mode).")

    # ---- TensorBoard (rank-0 only) ----
    writer = None
    if is_main(rank) and (not args.tb_disable):
        # Make a unique run directory per execution by default.
        if args.tb_run_name is None:
            base = os.path.splitext(os.path.basename(args.save_path))[0]
            stamp = time.strftime("%Y%m%d-%H%M%S")
            args.tb_run_name = f"{base}-{stamp}"
        tb_dir = os.path.join(args.tb_logdir, args.tb_run_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir, flush_secs=int(args.tb_flush_secs))

        # Log a compact config snapshot for reproducibility.
        try:
            cfg_lines = [f"{k}: {v}" for k, v in sorted(vars(args).items())]
            writer.add_text("config/args", "\n".join(cfg_lines), 0)
            writer.add_text("config/ddp", f"is_dist={is_dist} world_size={world} local_rank={local_rank}", 0)
        except Exception:
            # Never fail training because of TensorBoard I/O.
            pass
        print(f"[tensorboard] logdir -> {tb_dir}")

    ds = NPZProfileDataset(args.npz_dir)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if is_dist else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=build_collate_fn(tokenizer, max_tokens=args.seq_len),
        drop_last=True,  # stabilize shapes for DDP/all-reduce
    )

    model = StudentMamba2ForTracksAndMLM(
        vocab_size=vocab_size,
        n_tracks=args.n_tracks,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        d_conv=args.d_conv,
        dropout=args.dropout,
        pad_id=pad_id,
        downsample_factor=args.downsample_factor,
        crop_bins_each_side=args.crop_bins_each_side,
        output_activation=args.output_activation,
        use_moe=args.use_moe,
        use_moe_mamba=use_moe_mamba,
        use_moe_attn=use_moe_attn,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_ffn_mult=args.moe_ffn_mult,
        moe_dropout=args.moe_dropout,
        # moe_aux_weight=args.moe_aux_weight,
    ).to(device)

    # ---- Load Stage-1 checkpoint into backbone only (dense or MoE). ----
    ckpt_obj = torch.load(args.resume_path, map_location="cpu")
    state = _strip_module_prefix(_extract_state(ckpt_obj))

    # Detect whether the checkpoint already contains MoE parameters.
    has_moe_params = any(".moe." in k or ".router." in k for k in state.keys())

    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    if is_main(rank):
        print(f"[resume] loaded from {args.resume_path}")
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[resume] detected_moe_params={has_moe_params}")

    # If we enabled MoE but resumed from a dense checkpoint, perform Sparse Upcycling:
    # copy dense attention MLP into every MoE expert (ICLR'23 style).
    if args.use_moe and (not has_moe_params) and (not args.skip_upcycle_mlp_copy):
        upcycle_dense_attn_mlp_to_moe(model.backbone, verbose=is_main(rank))

    # Optionally save the upcycled MoE checkpoint (backbone only).
    if args.upcycle_ckpt_out is not None and is_main(rank):
        to_save_bb = model.backbone.state_dict()
        torch.save(
            {
                "model": to_save_bb,
                "config": vars(args),
                "note": "dense->moe upcycled backbone checkpoint",
            },
            args.upcycle_ckpt_out,
        )
        print(f"[upcycle] saved MoE backbone checkpoint -> {args.upcycle_ckpt_out}")

    if args.upcycle_only:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )

    total_steps = max(1, math.ceil(len(loader) / max(1, args.grad_accum)) * args.epochs)
    warmup = max(1, int(0.02 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # AMP setup
    amp_dtype = None
    scaler = None
    if args.amp == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    global_step = 0
    outer = trange(1, args.epochs + 1, disable=(not is_main(rank)), dynamic_ncols=True)

    for epoch in outer:
        if is_dist and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(loader, disable=(not is_main(rank)), leave=False, dynamic_ncols=True)
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)  # [B,T,C]

            # Prepare MLM-masked inputs.

            # Decide whether this micro-step should sync gradients (for grad accumulation).
            do_sync = (step % args.grad_accum == 0)

            is_ddp = dist.is_available() and dist.is_initialized()

            # In DDP:
            # - Always no_sync() for the first backward (profile) to avoid extra all-reduce.
            # - For the second backward (MLM), sync only when do_sync=True.
            # ctx_profile_sync = model.no_sync() if is_ddp else contextlib.nullcontext()
            ctx_profile_sync = (nullcontext() if (not is_ddp or do_sync) else model.no_sync())
            ctx_mlm_sync = (contextlib.nullcontext() if (not is_ddp or do_sync) else model.no_sync())

            autocast_ctx = torch.autocast("cuda", dtype=amp_dtype) if (
                        amp_dtype is not None and device.type == "cuda") else contextlib.nullcontext()

            # ----------------------------------------------------------------------
            # 1) Profile regression pass: UNMASKED input, NO logits, compute_pred=True
            # ----------------------------------------------------------------------
            with ctx_profile_sync:
                with autocast_ctx:
                    out_profile = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        compute_logits=False,
                        compute_pred=True,
                        logits_on_masked_only=False,
                    )
                    pred = out_profile["pred"]  # [B, T_pred, C]
                    bin_mask = out_profile.get("bin_mask", None)

                    # ---- Hard shape checks (T and C) ----
                    if pred.ndim != 3 or targets.ndim != 3:
                        raise ValueError(
                            f"Expected pred/targets rank-3, got pred={pred.shape}, targets={targets.shape}")
                    if pred.shape[0] != targets.shape[0]:
                        raise ValueError(f"Batch mismatch: pred={pred.shape}, targets={targets.shape}")
                    if pred.shape[1] != targets.shape[1]:
                        raise ValueError(
                            "Bin-length (T) mismatch. "
                            f"pred.T={pred.shape[1]} vs targets.T={targets.shape[1]}. "
                            "Check tokenizer stride, seq_len, downsample_factor, crop_bins_each_side, and NPZ bin_bp."
                        )
                    if pred.shape[2] != targets.shape[2]:
                        raise ValueError(
                            "Track/channel (C) mismatch. "
                            f"pred.C={pred.shape[2]} vs targets.C={targets.shape[2]}. "
                            "Check --n_tracks and NPZ y.shape[-1]."
                        )

                    # ---- Regression loss ----
                    if args.loss == "mse":
                        reg_loss = F.mse_loss(pred, targets, reduction="none")
                        if bin_mask is not None:
                            reg_loss = reg_loss * bin_mask.unsqueeze(-1).type_as(reg_loss)
                            denom = (bin_mask.sum() * targets.size(-1)).clamp_min(1.0)
                            reg_loss = reg_loss.sum() / denom
                        else:
                            reg_loss = reg_loss.mean()
                    else:
                        reg_loss = masked_smooth_l1(pred, targets, bin_mask)

                    # Scale for grad accumulation
                    reg_loss_scaled = reg_loss / max(1, args.grad_accum)

                # Backward immediately to free activations from profile graph.
                if scaler is not None:
                    scaler.scale(reg_loss_scaled).backward()
                else:
                    reg_loss_scaled.backward()

            # Explicitly drop large tensors ASAP (helps peak memory).
            del out_profile, pred, bin_mask

            # ----------------------------------------------------------------------
            # 2) MLM pass: MASKED input, compute_pred=False, logits_on_masked_only=True
            # ----------------------------------------------------------------------
            with ctx_mlm_sync:
                # Prepare MLM batch
                masked_input_ids, mlm_labels = make_mlm_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=mask_token_id,
                    vocab_size=vocab_size,
                    mlm_prob=args.mlm_prob,
                    mask_prob=args.mlm_mask_prob,
                    random_prob=args.mlm_random_prob,
                )

                with autocast_ctx:
                    out_mlm = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        compute_logits=False,  # do NOT build [B,L,V]
                        compute_pred=False,  # skip profile head to save memory
                        logits_on_masked_only=True,  # only masked positions
                        mlm_labels=mlm_labels,
                    )

                    logits_masked = out_mlm["logits_masked"]  # [N_mask, V]
                    labels_masked = out_mlm["labels_masked"]  # [N_mask]

                    if labels_masked.numel() == 0:
                        mlm_loss = logits_masked.sum() * 0.0  # safe zero
                    else:
                        mlm_loss = F.cross_entropy(
                            logits_masked,
                            labels_masked,
                            reduction="mean",
                        )

                    # MoE aux loss (load balancing) - attach once (MLM pass is fine)
                    moe_aux = out_mlm.get("moe_aux_loss", mlm_loss.new_zeros(()))
                    aux_term = (args.moe_aux_weight * moe_aux) if (
                                args.use_moe and args.moe_aux_weight > 0) else moe_aux.new_zeros(())

                    loss_val = reg_loss + args.mlm_weight * mlm_loss + aux_term
                    mlm_total_scaled = (args.mlm_weight * mlm_loss + aux_term) / max(1, args.grad_accum)

                if scaler is not None:
                    scaler.scale(mlm_total_scaled).backward()
                else:
                    mlm_total_scaled.backward()

            del out_mlm, masked_input_ids, mlm_labels, logits_masked, labels_masked

            # ----------------------------------------------------------------------
            # Optimizer step (only when do_sync / grad_accum boundary)
            # ----------------------------------------------------------------------
            if do_sync:
                grad_norm_val = None
                if scaler is not None:
                    scaler.unscale_(optim)
                    if grad_clip > 0:
                        grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optim.step()
                optim.zero_grad(set_to_none=True)

                sched.step()
                global_step += 1

                # ---- TensorBoard scalar logging (rank-0 only) ----
                if writer is not None and (global_step % max(1, int(args.tb_log_every)) == 0):
                    try:
                        writer.add_scalar("loss/total", float(loss_val.detach().float().item()), global_step)
                        writer.add_scalar("loss/reg", float(reg_loss.detach().float().item()), global_step)
                        writer.add_scalar("loss/mlm", float(mlm_loss.detach().float().item()), global_step)
                        writer.add_scalar("loss/moe_aux", float(aux_term.detach().float().item()), global_step)
                        writer.add_scalar("optim/lr", float(optim.param_groups[0]["lr"]), global_step)
                        writer.add_scalar("train/epoch", float(epoch), global_step)
                        if grad_norm_val is not None:
                            # clip_grad_norm_ returns a tensor-like scalar
                            writer.add_scalar("optim/grad_norm", float(grad_norm_val.detach().float().item()), global_step)
                    except Exception:
                        pass

            if is_main(rank):
                pbar.set_postfix(
                    {
                        "loss": f"{loss_val.detach().float().item():.4f}",
                        "reg": f"{reg_loss.detach().float().item():.4f}",
                        "mlm": f"{mlm_loss.detach().float().item():.4f}",
                        "aux": f"{aux_term.detach().float().item():.4f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                    }
                )

        if is_main(rank):
            to_save = model.module.state_dict() if is_dist else model.state_dict()
            torch.save({"model": to_save, "config": vars(args)}, args.save_path)
            outer.set_postfix({"saved": os.path.basename(args.save_path)})

            if writer is not None:
                try:
                    writer.flush()
                except Exception:
                    pass

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
