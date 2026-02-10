# train_student.py
# KD or MLM pretraining with multi-epoch progress bars, teacher micro-batching, and optional AMP.
# KD losses: embedding MSE, encoder MSE, masked KL. MLM loss: masked token cross-entropy.

import os
import math
import argparse
import random
import contextlib
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm.auto import tqdm, trange

from dataset import PositiveSeqDataset
from teacher import TeacherNT500M
from student import StudentMamba2
from torch.utils.tensorboard import SummaryWriter

# Optional throughput boost on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True


def setup_distributed_if_needed():
    """Init DDP if launched by torchrun; return (is_dist, rank, world_size, local_rank)."""
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def is_main_process(rank: int) -> bool:
    return rank == 0


def make_span_mask_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_rate: float,
    span_lambda: float
) -> torch.Tensor:
    """
    Return [B, M] positions per sample to be masked, with Poisson span length.
    We ensure positions are within valid tokens (attention_mask==1).
    """
    B, L = input_ids.size()
    all_pos_lists: List[List[int]] = []
    max_M = 0

    for b in range(B):
        valid = [i for i in range(L) if attention_mask[b, i].item() == 1]
        if not valid:
            all_pos_lists.append([0])
            max_M = max(max_M, 1)
            continue

        target_M = max(1, int(round(mask_rate * len(valid))))
        chosen, used = [], set()
        trials = 0
        while len(chosen) < target_M and trials < 20 * target_M:
            trials += 1
            start = random.choice(valid)
            span = max(1, torch.poisson(torch.tensor([span_lambda])).int().item())
            for k in range(span):
                pos = start + k
                if pos >= L or attention_mask[b, pos].item() == 0 or pos in used:
                    break
                used.add(pos)
                chosen.append(pos)
                if len(chosen) >= target_M:
                    break
        if not chosen:
            chosen = [valid[-1]]

        chosen = sorted(chosen)
        all_pos_lists.append(chosen)
        max_M = max(max_M, len(chosen))

    out = torch.zeros((B, max_M), dtype=torch.long, device=input_ids.device)
    for b, pos_list in enumerate(all_pos_lists):
        if len(pos_list) < max_M:
            pos_list = pos_list + [pos_list[-1]] * (max_M - len(pos_list))
        out[b] = torch.tensor(pos_list[:max_M], dtype=torch.long, device=input_ids.device)
    return out


def build_collate_fn(tokenizer, max_length: int):
    """
    Tokenize raw sequences with the teacher tokenizer to a fixed length.
    If you hit multiprocessing/pickling issues, set DataLoader(num_workers=0).
    """
    def _collate(batch_seqs: List[str]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].long(),
            "attention_mask": enc["attention_mask"].long()
        }
    return _collate


class AvgMeter:
    """Simple running average helper."""
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val: float, k: int = 1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)


def teacher_forward_microbatch(
    teacher: TeacherNT500M,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    masked_positions: torch.Tensor,
    mb: int,
    amp_dtype=None
) -> Dict[str, torch.Tensor]:
    """
    Run teacher in micro-batches to avoid OOM.
    Returns dict with:
      - input_embeds: [B, L, D_t]
      - last_hidden: [B, L, D_t]
      - masked_logits: [B, M, V] (M=masked positions count)
    """
    B = input_ids.size(0)
    outs = []
    for s in range(0, B, mb):
        e = min(B, s + mb)
        ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
        with ctx():
            o = teacher(
                input_ids=input_ids[s:e],
                attention_mask=attention_mask[s:e],
                masked_positions=masked_positions[s:e]
            )
        outs.append(o)

    def cat_or_first(key):
        vals = [o[key] for o in outs]
        return torch.cat(vals, dim=0) if len(vals) > 1 else vals[0]

    return {
        "input_embeds": cat_or_first("input_embeds"),
        "last_hidden": cat_or_first("last_hidden"),
        "masked_logits": cat_or_first("masked_logits"),
    }


def train_one_epoch(
    teacher: Optional[TeacherNT500M],
    student: StudentMamba2,
    rank: int,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mask_rate: float,
    span_lambda: float,
    tau: float,
    lambda_embed: float,
    lambda_hidden: float,
    lambda_kl: float,
    grad_accum_steps: int,
    epoch_idx: int,
    n_epochs: int,
    global_step_start: int,
    amp_dtype=None,
    teacher_microbatch: int = 2,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    writer=None,
    tb_log_interval: int = 10,
    pretrain_mode: str = "kd",
    mask_token_id: Optional[int] = None,
) -> Dict[str, float]:
    """
    One training epoch.

    pretrain_mode:
      - "kd": knowledge distillation from teacher (embed/hidden/kl losses).
      - "mlm": standalone masked language modeling (student only).
    """
    if pretrain_mode not in ("kd", "mlm"):
        raise ValueError(f"Unsupported pretrain_mode={pretrain_mode}")

    student.train()
    if pretrain_mode == "kd":
        meters = {k: AvgMeter() for k in ["loss", "embed", "hidden", "kl"]}
    else:
        meters = {k: AvgMeter() for k in ["loss", "mlm"]}

    global_step = global_step_start

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch_idx}/{n_epochs}",
        leave=False,
        dynamic_ncols=True,
        disable=(not is_main_process(rank))
    )

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, 1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # Compute masked positions once per batch
        masked_positions = make_span_mask_positions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_rate=mask_rate,
            span_lambda=span_lambda
        )

        # Teacher forward (KD mode only)
        t_out = None
        if pretrain_mode == "kd":
            assert teacher is not None, "Teacher must be provided in KD mode."
            t_out = teacher_forward_microbatch(
                teacher=teacher,
                input_ids=input_ids,
                attention_mask=attention_mask,
                masked_positions=masked_positions,
                mb=teacher_microbatch,
                amp_dtype=amp_dtype,
            )

        ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if amp_dtype else contextlib.nullcontext
        with ctx():
            if pretrain_mode == "kd":
                # KD mode: unmasked and masked student forwards
                s_out = student(input_ids=input_ids, attention_mask=attention_mask)
                masked_ids = teacher.mask_with_positions(input_ids, masked_positions)
                s_out_masked = student(input_ids=masked_ids, attention_mask=attention_mask)
                stu_obj = student.module if isinstance(student, DDP) else student
                kd = stu_obj.kd_losses(
                    student_out=s_out,
                    teacher_out=t_out,
                    masked_positions=masked_positions,
                    tau=tau,
                    lambda_embed=lambda_embed,
                    lambda_hidden=lambda_hidden,
                    lambda_kl=lambda_kl,
                    student_logits_masked=s_out_masked["logits"],
                )
                loss_total = kd["loss_total"]
                loss = loss_total / grad_accum_steps
                loss_mlm = None
            else:
                # MLM mode: student-only masked LM on masked_ids
                assert mask_token_id is not None, "mask_token_id is required for MLM mode."
                masked_ids = input_ids.clone()
                B, M = masked_positions.shape
                batch_idx = torch.arange(B, device=masked_ids.device).unsqueeze(-1).expand(B, M)
                masked_ids[batch_idx, masked_positions] = mask_token_id

                s_out = student(input_ids=masked_ids, attention_mask=attention_mask)
                logits = s_out["logits"]  # [B, L, V]

                logits_masked = StudentMamba2._gather_positions(logits, masked_positions)  # [B, M, V]
                Bm, Mm, V = logits_masked.shape
                logits_flat = logits_masked.reshape(Bm * Mm, V)

                target_ids = StudentMamba2._gather_positions(
                    input_ids.unsqueeze(-1), masked_positions
                ).squeeze(-1)  # [B, M]
                target_flat = target_ids.reshape(Bm * Mm)

                loss_mlm = F.cross_entropy(logits_flat, target_flat)
                loss = loss_mlm / grad_accum_steps

        # Backward and optimizer step with optional AMP
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0:
            # torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1


            if writer is not None and is_main_process(rank) and (global_step % tb_log_interval == 0):
                cur_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/lr", cur_lr, global_step)

                try:
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                except Exception:
                    pass
                if pretrain_mode == "kd":
                    writer.add_scalar("train/loss_total", float(loss_total.item()), global_step)
                    writer.add_scalar("train/loss_embed", float(kd["loss_embed"].item()), global_step)
                    writer.add_scalar("train/loss_hidden", float(kd["loss_hidden"].item()), global_step)
                    writer.add_scalar("train/loss_kl", float(kd["loss_kl"].item()), global_step)
                else:
                    writer.add_scalar("train/loss_mlm", float(loss_mlm.item()), global_step)

        # Update meters
        if pretrain_mode == "kd":
            meters["loss"].update(loss_total.item())
            meters["embed"].update(kd["loss_embed"].item())
            meters["hidden"].update(kd["loss_hidden"].item())
            meters["kl"].update(kd["loss_kl"].item())
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{meters['loss'].avg:.4f}",
                "emb": f"{meters['embed'].avg:.4f}",
                "hid": f"{meters['hidden'].avg:.4f}",
                "kl": f"{meters['kl'].avg:.4f}",
                "lr": f"{cur_lr:.2e}",
            })
        else:
            meters["loss"].update(loss_mlm.item())
            meters["mlm"].update(loss_mlm.item())
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{meters['loss'].avg:.4f}",
                "mlm": f"{meters['mlm'].avg:.4f}",
                "lr": f"{cur_lr:.2e}",
            })

    if pretrain_mode == "kd":
        return {
            "loss": meters["loss"].avg,
            "embed": meters["embed"].avg,
            "hidden": meters["hidden"].avg,
            "kl": meters["kl"].avg,
        }, global_step
    else:
        return {
            "loss": meters["loss"].avg,
            "mlm": meters["mlm"].avg,
        }, global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_dir", type=str, default="positive")
    parser.add_argument("--teacher_dir", type=str, default="src/hf/nucleotide-transformer-v2-500m-multi-species")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--mask_rate", type=float, default=0.30)
    parser.add_argument("--span_lambda", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lambda_embed", type=float, default=1.0)
    parser.add_argument("--lambda_hidden", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="student_mamba2.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        type=str,
        default="bf16",
        choices=["none", "bf16", "fp16"],
        help="Autocast dtype for teacher & student forward.",
    )
    parser.add_argument(
        "--teacher_microbatch",
        type=int,
        default=2,
        help="Process teacher forward in chunks of this batch size.",
    )
    parser.add_argument(
        "--pretrain_mode",
        type=str,
        default="kd",
        choices=["kd", "mlm"],
        help="Choose 'kd' for knowledge distillation or 'mlm' for standalone masked LM.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU ids to use, e.g. '0,2,3'. If set, CUDA_VISIBLE_DEVICES will be overwritten.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Optional checkpoint path to load student weights from before training.",
    )

    parser.add_argument("--tb_logdir", type=str, default="runs/train_student")
    parser.add_argument("--tb_log_interval", type=int, default=10)
    parser.add_argument("--tb_disable", action="store_true")
    parser.add_argument(
        "--tb_run_name", type=str, default=None,
        help = "Optional run name subfolder under tb_logdir.")


    args = parser.parse_args()

    # Optional GPU selection: must happen before any CUDA / distributed initialization
    if args.gpu_ids is not None:
        # This affects which physical GPUs are visible as cuda:0,1,2,...
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_dist, rank, world_size, local_rank = setup_distributed_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    writer = None

    if is_main_process(rank) and (not args.tb_disable) and (SummaryWriter is not None):
        run_name = args.tb_run_name
        if run_name is None:
            run_name = f"{args.pretrain_mode}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}"
        tb_dir = os.path.join(args.tb_logdir, run_name)
        writer = SummaryWriter(log_dir=tb_dir)
        try:
            writer.add_text("config/args", str(vars(args)), 0)
        except Exception:
            pass
        print(f"[tb] logdir: {tb_dir}")

    # Data
    ds = PositiveSeqDataset(args.positive_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No sequences found under: {args.positive_dir}")
    if is_main_process(rank):
        print(f"[data] sequences: {len(ds)}")

    # Tokenizer (teacher's)
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_dir, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Teacher tokenizer has no pad_token_id; please set one before training.")
    pad_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id

    if args.pretrain_mode == "mlm" and mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id; MLM pretraining requires a [MASK] token.")

    if is_main_process(rank):
        print(f"[tok] vocab_size={tokenizer.vocab_size}, pad_id={pad_id}, mask_id={mask_token_id}")

    collate = build_collate_fn(tokenizer, max_length=args.seq_len)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        sampler=sampler,
    )

    # Teacher (only used in KD mode)
    teacher: Optional[TeacherNT500M] = None
    if args.pretrain_mode == "kd":
        teacher = TeacherNT500M(args.teacher_dir, device=str(device))

    # Student (create on CPU first)
    student = StudentMamba2(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        d_conv=4,
        dropout=0.0,
        tie_lm_head=False,
        pad_id=pad_id,
    )

    # For KD mode, initialize projection heads before loading checkpoint so that
    # proj_to_teacher_* weights (if present) can be restored.
    if args.pretrain_mode == "kd":
        student.set_teacher_dims(
            d_teacher_embed=teacher.teacher_embed_dim,
            d_teacher_hidden=teacher.teacher_hidden_dim,
        )

    # Optional resume: load checkpoint weights into student BEFORE DDP wrapping
    if args.resume_path is not None:
        if is_main_process(rank):
            print(f"[resume] Loading student checkpoint from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location="cpu")
        # Support both {"model": state_dict} and plain state_dict
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = student.load_state_dict(state_dict, strict=False)
        if is_main_process(rank):
            if missing:
                print(f"[resume] Missing keys in loaded state_dict: {len(missing)}")
            if unexpected:
                print(f"[resume] Unexpected keys in loaded state_dict: {len(unexpected)}")

    # Move to device and wrap with DDP if needed
    if is_dist:
        student = DDP(
            student.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        student = student.to(device)

    # Optimizer + cosine schedule with 2% warmup (per-step)
    optim = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    total_steps = max(1, math.ceil(len(loader) / max(1, args.grad_accum)) * args.epochs)
    warmup = max(1, int(0.02 * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    # AMP selection
    amp_dtype = None
    scaler = None
    if args.amp == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif args.amp == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        amp_dtype = None
        scaler = None

    # Train loop (note: this "resumes" only weights; optimizer/scheduler start fresh)
    global_step = 0
    outer = trange(
        1,
        args.epochs + 1,
        desc="Epochs",
        leave=True,
        dynamic_ncols=True,
        disable=(not is_main_process(rank)),
    )
    for epoch in outer:
        if is_dist and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        stats, global_step = train_one_epoch(
            teacher=teacher,
            student=student,
            rank=rank,
            loader=loader,
            optimizer=optim,
            scheduler=scheduler,
            device=device,
            mask_rate=args.mask_rate,
            span_lambda=args.span_lambda,
            tau=args.tau,
            lambda_embed=args.lambda_embed,
            lambda_hidden=args.lambda_hidden,
            lambda_kl=args.lambda_kl,
            grad_accum_steps=args.grad_accum,
            epoch_idx=epoch,
            n_epochs=args.epochs,
            global_step_start=global_step,
            amp_dtype=amp_dtype,
            teacher_microbatch=args.teacher_microbatch,
            scaler=scaler,
            writer = writer,
            tb_log_interval = args.tb_log_interval,
            pretrain_mode=args.pretrain_mode,
            mask_token_id=mask_token_id,
        )

        if args.pretrain_mode == "kd":
            outer.set_postfix({
                "loss": f"{stats['loss']:.4f}",
                "emb": f"{stats['embed']:.4f}",
                "hid": f"{stats['hidden']:.4f}",
                "kl": f"{stats['kl']:.4f}",
            })
        else:
            outer.set_postfix({
                "loss": f"{stats['loss']:.4f}",
                "mlm": f"{stats['mlm']:.4f}",
            })

        # Save checkpoint after each epoch
        if is_main_process(rank):
            ckpt = {
                "model": (student.module.state_dict() if is_dist else student.state_dict()),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "d_state": args.d_state,
                    "expand": args.expand,
                    "seq_len": args.seq_len,
                    "pad_id": pad_id,
                    "teacher_dir": args.teacher_dir,
                    "pretrain_mode": args.pretrain_mode,
                },
            }
            torch.save(ckpt, args.save_path)

    if is_main_process(rank):
        print(f"Saved final checkpoint: {args.save_path}")

        if writer is not None:
            writer.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
