#!/usr/bin/env python
"""
Evaluate encoders on the GUE benchmark (DNABERT-2 GUE).

Dataset layout (local, cloned or snapshot-downloaded from HF):
  <datasets_root>/
    prom_core_all/
      train.csv
      dev.csv
      test.csv
    splice_reconstructed/
      train.csv
      dev.csv
      test.csv
    EPI_GM12878/
      train.csv
      dev.csv
      test.csv
    ...

CSV schema varies across subsets:
  - Most subsets: one sequence column (e.g., "sequence" / "seq") + "label"
  - EPI subsets: two sequence columns (e.g., "enhancer" and "promoter") + "label"
This script auto-detects label/sequence columns and supports paired inputs by feeding
(text, text_pair) to the tokenizer (recommended).

Selection & reporting:
  - Select best checkpoint by DEV MCC
  - Report DEV MCC/macro-F1 and TEST MCC/macro-F1

Multi-GPU evaluation:
  - Launch with torchrun; each rank evaluates a shard of tasks and rank0 merges JSON.

Example:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    -m src.eval.eval_GUE_benchmark \
    --encoder_type student \
    --student_ckpt config/student_stage1_mlm_1.pt \
    --student_tokenizer hf/nucleotide-transformer-v2-250m-multi-species \
    --datasets_root data/benchmark/GUE \
    --datasets auto \
    --mode full_finetune \
    --max_len 512 --batch_size 8 --epochs 10 --lr 2e-5 --amp \
    --metrics_out src/eval/results/student_gue.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torch import amp as _torch_amp
from tqdm.auto import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM


# -------------------------
# Distributed helpers
# -------------------------

def setup_distributed_if_needed() -> Tuple[bool, int, int, int]:
    """Init DDP if launched by torchrun; return (is_dist, rank, world_size, local_rank)."""
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def is_main_process(rank: int) -> bool:
    return rank == 0


def dist_all_gather_object(obj: Any, world_size: int) -> List[Any]:
    """Gather python objects from all ranks."""
    if world_size == 1:
        return [obj]
    out: List[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# CSV utilities
# -------------------------

def _read_csv_any(path: Path):
    """Read a CSV file into a pandas DataFrame."""
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read csv: {path}\nReason: {type(e).__name__}: {e}") from e


def find_candidate_gue_tasks(root: Path) -> List[Path]:
    """Find immediate subdirectories that contain train.csv, dev.csv, test.csv."""
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "train.csv").exists() and (p / "dev.csv").exists() and (p / "test.csv").exists():
            out.append(p)
    return out


def _pick_label_col(df) -> str:
    """Heuristically pick the label column."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for k in ["label", "labels", "y", "target", "class"]:
        if k in lower:
            return lower[k]
    # fallback: last numeric column
    for c in cols[::-1]:
        if str(df[c].dtype).startswith(("int", "float")):
            return c
    raise KeyError(f"Cannot find label column in columns: {cols}")


def _pick_sequence_cols(df, label_col: str) -> List[str]:
    """Heuristically pick one or two sequence columns (strings)."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # Common single-seq names
    for k in ["sequence", "seq", "dna", "text"]:
        if k in lower and lower[k] != label_col:
            return [lower[k]]

    # Common EPI pair names
    if "enhancer" in lower and "promoter" in lower:
        c1, c2 = lower["enhancer"], lower["promoter"]
        if c1 != label_col and c2 != label_col:
            return [c1, c2]

    # Fallback: first 1-2 string/object columns excluding label
    import pandas as pd
    cand: List[str] = []
    for c in cols:
        if c == label_col:
            continue
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            cand.append(c)

    if len(cand) == 0:
        raise KeyError(f"Cannot find any string sequence column (excluding label={label_col}) in columns: {cols}")
    if len(cand) == 1:
        return [cand[0]]
    return cand[:2]


def _sanitize_dna(seq: str) -> str:
    """Keep only A/C/G/T/N (uppercase)."""
    s = str(seq).upper().replace(" ", "").replace("\t", "")
    return "".join(ch for ch in s if ch in "ACGTN")


@dataclass
class GUESplit:
    train_df: Any
    dev_df: Any
    test_df: Any
    label_col: str
    seq_cols: List[str]


def load_gue_task(task_dir: Path) -> GUESplit:
    """Load train/dev/test DataFrames and auto-detect label/sequence columns."""
    df_tr = _read_csv_any(task_dir / "train.csv")
    df_de = _read_csv_any(task_dir / "dev.csv")
    df_te = _read_csv_any(task_dir / "test.csv")

    label_col = _pick_label_col(df_tr)
    seq_cols = _pick_sequence_cols(df_tr, label_col=label_col)

    keep = seq_cols + [label_col]
    df_tr = df_tr[keep].dropna().reset_index(drop=True)
    df_de = df_de[keep].dropna().reset_index(drop=True)
    df_te = df_te[keep].dropna().reset_index(drop=True)

    return GUESplit(df_tr, df_de, df_te, label_col=label_col, seq_cols=seq_cols)


# -------------------------
# Dataset wrapper
# -------------------------

class GUEDataset(Dataset):
    """Holds (seq1, seq2_or_None, label_int)."""

    def __init__(self, seq1: List[str], seq2: Optional[List[str]], labels: List[int]):
        assert len(seq1) == len(labels)
        if seq2 is not None:
            assert len(seq2) == len(labels)
        self.seq1 = [str(s) for s in seq1]
        self.seq2 = [str(s) for s in seq2] if seq2 is not None else None
        self.labels = [int(y) for y in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if self.seq2 is None:
            return self.seq1[i], None, self.labels[i]
        return self.seq1[i], self.seq2[i], self.labels[i]


def build_datasets_from_split(split: GUESplit) -> Tuple[GUEDataset, GUEDataset, GUEDataset, Dict[Any, int]]:
    """Build train/dev/test datasets with consistent label remapping across all splits."""
    lab_tr = split.train_df[split.label_col].tolist()
    lab_de = split.dev_df[split.label_col].tolist()
    lab_te = split.test_df[split.label_col].tolist()

    uniq = sorted(set([int(x) for x in lab_tr + lab_de + lab_te]))
    lab2id = {lab: i for i, lab in enumerate(uniq)}

    def map_labels(arr):
        return [lab2id[int(x)] for x in arr]

    sc = split.seq_cols
    if len(sc) == 1:
        s_tr = [_sanitize_dna(s) for s in split.train_df[sc[0]].tolist()]
        s_de = [_sanitize_dna(s) for s in split.dev_df[sc[0]].tolist()]
        s_te = [_sanitize_dna(s) for s in split.test_df[sc[0]].tolist()]
        ds_tr = GUEDataset(s_tr, None, map_labels(lab_tr))
        ds_de = GUEDataset(s_de, None, map_labels(lab_de))
        ds_te = GUEDataset(s_te, None, map_labels(lab_te))
    else:
        s1_tr = [_sanitize_dna(s) for s in split.train_df[sc[0]].tolist()]
        s2_tr = [_sanitize_dna(s) for s in split.train_df[sc[1]].tolist()]
        s1_de = [_sanitize_dna(s) for s in split.dev_df[sc[0]].tolist()]
        s2_de = [_sanitize_dna(s) for s in split.dev_df[sc[1]].tolist()]
        s1_te = [_sanitize_dna(s) for s in split.test_df[sc[0]].tolist()]
        s2_te = [_sanitize_dna(s) for s in split.test_df[sc[1]].tolist()]
        ds_tr = GUEDataset(s1_tr, s2_tr, map_labels(lab_tr))
        ds_de = GUEDataset(s1_de, s2_de, map_labels(lab_de))
        ds_te = GUEDataset(s1_te, s2_te, map_labels(lab_te))

    return ds_tr, ds_de, ds_te, lab2id


# -------------------------
# Metrics (MCC + macro-F1)
# -------------------------

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()


def precision_recall_f1_macro(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> Tuple[float, float, float]:
    precs, recs, f1s = [], [], []
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return float(sum(precs) / n_classes), float(sum(recs) / n_classes), float(sum(f1s) / n_classes)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int) -> List[List[int]]:
    cm = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t)][int(p)] += 1
    return cm


def mcc_from_confusion(cm: List[List[int]]) -> float:
    """Multi-class MCC (Gorodkin)."""
    K = len(cm)
    if K == 0:
        return 0.0
    s = sum(sum(row) for row in cm)
    c = sum(cm[i][i] for i in range(K))
    p = [sum(cm[k][j] for j in range(K)) for k in range(K)]
    t = [sum(cm[i][k] for i in range(K)) for k in range(K)]
    sum_p2 = sum(x * x for x in p)
    sum_t2 = sum(x * x for x in t)
    denom = math.sqrt(max((s * s - sum_t2), 0.0) * max((s * s - sum_p2), 0.0))
    if denom == 0:
        return 0.0
    num = c * s - sum(p[k] * t[k] for k in range(K))
    return float(num / denom)


def metrics_from_logits(y_true: torch.Tensor, logits: torch.Tensor, n_classes: int) -> Tuple[Dict[str, float], List[List[int]]]:
    """
    Avoid CPU softmax(fp16/bf16) pitfalls by using argmax only.
    """
    if logits.device.type == "cpu" and logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    y_pred = torch.argmax(logits, dim=-1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1_macro(y_true, y_pred, n_classes)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    mcc = mcc_from_confusion(cm)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc}, cm


# -------------------------
# Encoders
# -------------------------

class BaseEncoder(nn.Module):
    def encode(self, seqs1: List[str], seqs2: Optional[List[str]], max_len: int, amp: bool = False) -> torch.Tensor:
        raise NotImplementedError


class _HFMeanPoolMixin:
    def _mean_pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            mask = torch.ones(hidden.size(0), hidden.size(1), device=hidden.device, dtype=hidden.dtype)
        else:
            mask = attention_mask.to(hidden.device, dtype=hidden.dtype)
        mask = mask.unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class HFEncoder(BaseEncoder, _HFMeanPoolMixin):
    def __init__(self, model_path: str, device: torch.device, local_only: bool = False, prefer_mlm: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=local_only)

        model = None
        if prefer_mlm:
            try:
                model = AutoModelForMaskedLM.from_pretrained(
                    model_path, config=self.config, trust_remote_code=True, local_files_only=local_only
                )
            except Exception:
                model = None
        if model is None:
            model = AutoModel.from_pretrained(
                model_path, config=self.config, trust_remote_code=True, local_files_only=local_only
            )

        self.model = model.to(device)
        self.device = device

    def encode(self, seqs1: List[str], seqs2: Optional[List[str]], max_len: int, amp: bool = False) -> torch.Tensor:
        # Use native pair encoding (adds [SEP] correctly if the tokenizer supports it).
        if seqs2 is None:
            batch = self.tokenizer(
                seqs1, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt", add_special_tokens=True
            )
        else:
            batch = self.tokenizer(
                seqs1, seqs2, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt", add_special_tokens=True
            )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with _torch_amp.autocast("cuda", enabled=amp):
            out = self.model(**batch, output_hidden_states=True, return_dict=True)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else out[0]
            pooled = self._mean_pool(hidden, batch.get("attention_mask", None))
        return pooled


class StudentEncoder(BaseEncoder):
    """
    Student encoder wrapper.

    Supports both dense and MoE student checkpoints by importing StudentMamba2 from `model_FRE.py`
    (preferred) or falling back to the dense implementation if needed.

    Checkpoint formats supported:
      - raw state_dict
      - {"model": state_dict, "config": {...}}
      - {"state_dict": state_dict, "dense_cfg": {...}, "moe_cfg": {...}}
    """

    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        device: torch.device,
        local_only: bool = False,
        force_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
        moe_aux_weight: float = 1e-2,
        verbose: bool = True,
    ):
        super().__init__()

        # -------- Robust import across possible project layouts --------
        StudentMamba2 = None
        tried = []
        for mod in ["src.student_moe", "src.models.student_moe", "student_moe", "models.student_moe",
                    "src.student", "src.models.student", "student", "models.student"]:
            try:
                m = __import__(mod, fromlist=["StudentMamba2"])
                StudentMamba2 = getattr(m, "StudentMamba2")
                break
            except Exception as e:
                tried.append(f"{mod}: {type(e).__name__}")
                continue
        if StudentMamba2 is None:
            raise ImportError(
                "Cannot import StudentMamba2. Tried:\n  - " + "\n  - ".join(tried)
            )

        # -------- Tokenizer --------
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=local_only
        )
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.device = device

        # -------- Checkpoint parsing --------
        ckpt = torch.loaFd(ckpt_path, map_location="cpu")

        def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
            if isinstance(obj, dict):
                if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                    return obj["state_dict"]
                if "model" in obj and isinstance(obj["model"], dict):
                    return obj["model"]
            # raw state_dict
            if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
                return obj  # type: ignore
            raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

        def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not sd:
                return sd
            if any(k.startswith("module.") for k in sd.keys()):
                return {k.replace("module.", "", 1): v for k, v in sd.items()}
            return sd

        def _maybe_strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            """If most keys start with a prefix (e.g., 'backbone.'), strip it for encoder-only loading."""
            if not sd:
                return sd
            keys = list(sd.keys())
            n_pref = sum(1 for k in keys if k.startswith(prefix))
            # Heuristic: if >= 60% keys share the prefix, assume it's a wrapper module checkpoint.
            if n_pref >= int(0.6 * len(keys)):
                return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            return sd

        state = _strip_module_prefix(_extract_state_dict(ckpt))
        # Common stage-2 checkpoints save the encoder under 'backbone.'.
        state = _maybe_strip_prefix(state, "backbone.")
        # Some training wrappers nest backbone twice.
        state = _maybe_strip_prefix(state, "backbone.")

        # Config (dense_cfg has priority for upcycled checkpoints)
        cfg = {}
        if isinstance(ckpt, dict):
            cfg = ckpt.get("dense_cfg") or ckpt.get("config") or ckpt.get("student_core_config") or {}

        d_model = int(cfg.get("d_model", 384))
        n_layers = int(cfg.get("n_layers", 24))
        d_state = int(cfg.get("d_state", 16))
        expand = int(cfg.get("expand", 2))
        d_conv = int(cfg.get("d_conv", cfg.get("dconv", 4)))

        # Detect MoE from checkpoint keys or metadata
        moe_cfg = {}
        if isinstance(ckpt, dict):
            moe_cfg = ckpt.get("moe_cfg") or {}

        def _looks_like_moe(sd: Dict[str, torch.Tensor]) -> bool:
            # Heuristic: keys introduced by MoEFeedForward
            for k in sd.keys():
                if ".moe." in k or ".router." in k or "router.proj" in k or ".experts." in k:
                    return True
            return False

        # use_moe = bool(force_moe or _looks_like_moe(state) or moe_cfg.get("use_moe", False))

        import re  # add at file top if not present

        # ---- Infer MoE placement from checkpoint keys ----
        # StudentMamba2 layout: first M blocks are Mamba2MVBlock, last K blocks are AttnBlock1D
        K = max(0, n_layers // 4)
        M = n_layers - K

        moe_block_ids = set()
        for k in state.keys():
            if (".moe." in k) or (".router." in k) or (".experts." in k) or ("router.proj" in k):
                m = re.match(r"^blocks\.(\d+)\.", k)
                if m is not None:
                    moe_block_ids.add(int(m.group(1)))

        # Prefer explicit moe_cfg if present; otherwise infer from moe_block_ids.
        use_moe_mamba = moe_cfg.get("use_moe_mamba", None)
        use_moe_attn = moe_cfg.get("use_moe_attn", None)

        if use_moe_mamba is None and use_moe_attn is None:
            # If checkpoint only has MoE in attention blocks, this will set:
            #   use_moe_mamba=False, use_moe_attn=True
            use_moe_mamba = any(i < M for i in moe_block_ids)
            use_moe_attn = any(i >= M for i in moe_block_ids)

        # If user forces MoE but ckpt doesn't contain any MoE params, default to attention-only MoE.
        # (This matches your stage-2 design and avoids creating MoE in early Mamba blocks.)
        if force_moe and (not moe_block_ids) and (use_moe_mamba is None) and (use_moe_attn is None):
            use_moe_mamba = False
            use_moe_attn = True

        # Final global flag: true if any block family uses MoE.
        use_moe = bool(use_moe_mamba or use_moe_attn or force_moe)

        # If moe_cfg exists, prefer it for hyperparams
        if moe_cfg:
            moe_num_experts = int(moe_cfg.get("moe_num_experts", moe_cfg.get("num_experts", moe_num_experts)))
            moe_top_k = int(moe_cfg.get("moe_top_k", moe_cfg.get("top_k", moe_top_k)))
            moe_ffn_mult = float(moe_cfg.get("moe_ffn_mult", moe_cfg.get("ffn_mult", moe_ffn_mult)))
            moe_dropout = float(moe_cfg.get("moe_dropout", moe_cfg.get("dropout", moe_dropout)))
            moe_aux_weight = float(moe_cfg.get("moe_aux_weight", moe_cfg.get("aux_weight", moe_aux_weight)))

        # Build kwargs and filter by signature for compatibility across versions.
        import inspect
        sig = inspect.signature(StudentMamba2.__init__)
        kwargs = dict(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            pad_id=self.pad_id,
            # MoE args (only used if supported)
            use_moe=use_moe,
            use_moe_mamba=use_moe_mamba,
            use_moe_attn=use_moe_attn,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_ffn_mult=moe_ffn_mult,
            moe_dropout=moe_dropout,
            moe_aux_weight=moe_aux_weight,
        )

        print(
            f"[student{'+moe' if use_moe else ''}] "
            f"use_moe={use_moe} use_moe_mamba={use_moe_mamba} use_moe_attn={use_moe_attn} "
            f"d_model={d_model} n_layers={n_layers} expand={expand} d_state={d_state} d_conv={d_conv}"
        )

        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        self.model = StudentMamba2(**kwargs)

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.model.to(device)
        self.model.eval()

        if verbose:
            msg = f"[student] use_moe={use_moe} d_model={d_model} n_layers={n_layers} expand={expand} d_state={d_state} d_conv={d_conv}"
            print(msg)
            if missing:
                print(f"[student][warn] Missing keys while loading: {len(missing)}")
            if unexpected:
                print(f"[student][warn] Unexpected keys while loading: {len(unexpected)}")

    def encode(self, seqs1: List[str], seqs2: Optional[List[str]], max_len: int, amp: bool = False) -> torch.Tensor:
        if seqs2 is None:
            batch = self.tokenizer(
                seqs1, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt", add_special_tokens=True
            )
        else:
            batch = self.tokenizer(
                seqs1, seqs2, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt", add_special_tokens=True
            )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with _torch_amp.autocast("cuda", enabled=amp):
            # IMPORTANT: classification does not need LM logits
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                compute_logits=False,
            )
            pooled = out["pooled"]
        return pooled


def build_encoder(
    model_path: Optional[str],
    device: torch.device,
    encoder_type: str,
    local_only: bool,
    student_ckpt: Optional[str],
    student_tokenizer: Optional[str],
    # MoE knobs (used only for student/student_moe if ckpt does not provide moe_cfg)
    moe_num_experts: int = 8,
    moe_top_k: int = 2,
    moe_ffn_mult: float = 4.0,
    moe_dropout: float = 0.0,
    moe_aux_weight: float = 1e-2,
) -> BaseEncoder:
    """
    Build an encoder instance.

    encoder_type:
      - "auto": if student_ckpt is provided, use student; otherwise use HF encoder.
      - "student": load student checkpoint; auto-detect MoE from ckpt.
      - "student_moe": force MoE mode when constructing student model.
    """
    # Auto mode
    if encoder_type == "auto":
        if student_ckpt and student_tokenizer:
            encoder_type = "student"
        else:
            encoder_type = "hf"

    if encoder_type in ("student", "student_moe"):
        if not student_ckpt or not student_tokenizer:
            raise ValueError("--encoder_type student/student_moe requires --student_ckpt and --student_tokenizer")

        verbose = True
        try:
            if dist.is_available() and dist.is_initialized():
                verbose = (dist.get_rank() == 0)
        except Exception:
            verbose = True

        return StudentEncoder(
            ckpt_path=student_ckpt,
            tokenizer_path=student_tokenizer,
            device=device,
            local_only=local_only,
            force_moe=(encoder_type == "student_moe"),
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_ffn_mult=moe_ffn_mult,
            moe_dropout=moe_dropout,
            moe_aux_weight=moe_aux_weight,
            verbose=verbose,
        )

    # HF encoder fallback
    if not model_path:
        raise ValueError("--model_path is required unless --encoder_type=student/student_moe")
    return HFEncoder(model_path, device=device, local_only=local_only, prefer_mlm=True)


# -------------------------
# LoRA


# -------------------------

class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: y = Wx + (BAx) * (alpha/r)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / max(r, 1)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros((r, base.in_features), device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros((base.out_features, r), device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x_ = self.drop(x)
        lora_w = (self.B @ self.A).to(x_.dtype)
        return y + F.linear(x_, lora_w) * self.scale


def _should_inject(module_name: str, include: List[str], exclude: List[str]) -> bool:
    if include and not any(k in module_name for k in include):
        return False
    if exclude and any(k in module_name for k in exclude):
        return False
    return True


def inject_lora(
    model: nn.Module,
    include_kw: List[str],
    exclude_kw: List[str],
    r: int,
    alpha: int,
    dropout: float,
    verbose: bool = True,
) -> int:
    cnt = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _should_inject(name, include_kw, exclude_kw):
            parent = model
            *prefix, last = name.split(".")
            for p in prefix:
                parent = getattr(parent, p)
            setattr(parent, last, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            cnt += 1
    if verbose:
        inc = include_kw or ["<ALL>"]
        exc = exclude_kw or ["<NONE>"]
        print(f"[LoRA] Injected into {cnt} Linear modules (include={inc}, exclude={exc})")
    return cnt


def reset_lora_params(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.kaiming_uniform_(m.A, a=math.sqrt(5))
            nn.init.zeros_(m.B)


# -------------------------
# Training helpers
# -------------------------

class EncoderWithHead(nn.Module):
    def __init__(self, enc: BaseEncoder, in_dim: int, n_classes: int, amp: bool):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(in_dim, n_classes)
        self.amp = amp

    def forward(self, seqs1: List[str], seqs2: Optional[List[str]], max_len: int) -> torch.Tensor:
        with _torch_amp.autocast("cuda", enabled=self.amp):
            z = self.enc.encode(seqs1, seqs2, max_len=max_len, amp=self.amp)
            return self.head(z)


def iterate_minibatches(ds: Dataset, batch_size: int) -> Iterable[Tuple[List[str], Optional[List[str]], torch.Tensor]]:
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        s1 = [x[0] for x in batch]
        s2_list = [x[1] for x in batch]
        s2 = None if all(v is None for v in s2_list) else s2_list
        y = torch.tensor([x[2] for x in batch], dtype=torch.long)
        yield s1, s2, y


@torch.no_grad()
def eval_on(
    model: EncoderWithHead,
    ds: Dataset,
    max_len: int,
    batch_size: int,
    amp: bool,
    device: torch.device,
    n_classes: int,
) -> Dict[str, float]:
    model.eval()
    logits_all, y_all = [], []
    for s1, s2, y in iterate_minibatches(ds, batch_size):
        y_all.append(y)
        with _torch_amp.autocast("cuda", enabled=amp):
            logits = model(s1, s2, max_len=max_len)
        logits_all.append(logits.detach().cpu())
    logits = torch.cat(logits_all, 0)
    y_true = torch.cat(y_all, 0)
    return metrics_from_logits(y_true, logits, n_classes)[0]


def run_linear_probe(
    enc: BaseEncoder,
    ds_train: Dataset,
    ds_dev: Dataset,
    ds_test: Dataset,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    device: torch.device,
    patience: int,
) -> Dict[str, Any]:
    """Embed once, train a linear head, select by dev MCC."""
    def embed_split(ds: Dataset, desc: str) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, labs = [], []
        for s1, s2, y in tqdm(iterate_minibatches(ds, batch_size), desc=desc, leave=False):
            z = enc.encode(s1, s2, max_len=max_len, amp=amp).detach().cpu()
            feats.append(z)
            labs.append(y)
        return torch.cat(feats, 0), torch.cat(labs, 0)

    Xtr, ytr = embed_split(ds_train, "Embed train")
    Xde, yde = embed_split(ds_dev, "Embed dev")
    Xte, yte = embed_split(ds_test, "Embed test")

    n_classes = int(max(ytr.max(), yde.max(), yte.max()).item()) + 1
    head = nn.Linear(Xtr.size(1), n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_mcc, best_state, wait = -1e9, None, 0

    for _ in tqdm(range(1, epochs + 1), desc="LinearProbe", leave=False):
        head.train()
        order = torch.randperm(Xtr.size(0))
        for i in range(0, Xtr.size(0), batch_size):
            idx = order[i:i + batch_size]
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

        # Dev MCC selection
        head.eval()
        dev_logits = []
        with torch.no_grad():
            for i in range(0, Xde.size(0), batch_size):
                dev_logits.append(head(Xde[i:i + batch_size].to(device)).cpu())
        m_dev, _ = metrics_from_logits(yde, torch.cat(dev_logits, 0), n_classes)

        if m_dev["mcc"] > best_mcc:
            best_mcc = m_dev["mcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    # Final dev/test
    head.eval()
    with torch.no_grad():
        dev_logits = []
        for i in range(0, Xde.size(0), batch_size):
            dev_logits.append(head(Xde[i:i + batch_size].to(device)).cpu())
        m_dev, _ = metrics_from_logits(yde, torch.cat(dev_logits, 0), n_classes)

        test_logits = []
        for i in range(0, Xte.size(0), batch_size):
            test_logits.append(head(Xte[i:i + batch_size].to(device)).cpu())
        m_test, _ = metrics_from_logits(yte, torch.cat(test_logits, 0), n_classes)

    return {"dev": m_dev, "test": m_test, "select_metric": "mcc"}


def run_lora(
    enc: BaseEncoder,
    ds_train: Dataset,
    ds_dev: Dataset,
    ds_test: Dataset,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    device: torch.device,
    include_kw: List[str],
    exclude_kw: List[str],
    r: int,
    alpha: int,
    dropout: float,
    patience: int,
) -> Dict[str, Any]:
    """Train LoRA adapters + head, select by dev MCC."""
    base_model = getattr(enc, "model", None)
    if base_model is None:
        raise RuntimeError("Encoder must expose .model for LoRA injection.")

    # Freeze base weights first
    for p in base_model.parameters():
        p.requires_grad_(False)

    if not exclude_kw:
        exclude_kw = ["embedding", "embeddings", "norm", "layernorm"]

    inject_lora(base_model, include_kw, exclude_kw, r=r, alpha=alpha, dropout=dropout, verbose=True)
    reset_lora_params(base_model)
    base_model.to(device)

    # Probe feature dim and n_classes
    with torch.no_grad():
        z = enc.encode(["ACGT"], None, max_len=max_len, amp=amp)
    in_dim = int(z.shape[-1])
    labels_all = [y for _, _, y in ds_train] + [y for _, _, y in ds_dev] + [y for _, _, y in ds_test]
    n_classes = int(max(labels_all)) + 1

    model = EncoderWithHead(enc, in_dim, n_classes=n_classes, amp=amp).to(device)
    trainable = [p for p in base_model.parameters() if p.requires_grad] + list(model.head.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_mcc, best_state, wait = -1e9, None, 0

    for _ in tqdm(range(1, epochs + 1), desc="LoRA", leave=False):
        base_model.train()
        model.train()

        order = torch.randperm(len(ds_train)).tolist()
        for i0 in range(0, len(order), batch_size):
            idx = order[i0:i0 + batch_size]
            batch = [ds_train[ii] for ii in idx]
            s1 = [b[0] for b in batch]
            s2_list = [b[1] for b in batch]
            s2 = None if all(v is None for v in s2_list) else s2_list
            y = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)

            with _torch_amp.autocast("cuda", enabled=amp):
                logits = model(s1, s2, max_len=max_len)
                loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        m_dev = eval_on(model, ds_dev, max_len, batch_size, amp, device, n_classes)
        if m_dev["mcc"] > best_mcc:
            best_mcc = m_dev["mcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    m_dev = eval_on(model, ds_dev, max_len, batch_size, amp, device, n_classes)
    m_test = eval_on(model, ds_test, max_len, batch_size, amp, device, n_classes)
    return {"dev": m_dev, "test": m_test, "select_metric": "mcc"}


def run_full_finetune(
    enc: BaseEncoder,
    ds_train: Dataset,
    ds_dev: Dataset,
    ds_test: Dataset,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    device: torch.device,
    patience: int,
) -> Dict[str, Any]:
    """Fine-tune encoder + head, select by dev MCC."""
    base_model = getattr(enc, "model", None)
    if base_model is None:
        raise RuntimeError("Encoder must expose .model for full finetune.")

    for p in base_model.parameters():
        p.requires_grad_(True)

    # Probe feature dim and n_classes
    with torch.no_grad():
        z = enc.encode(["ACGT"], None, max_len=max_len, amp=amp)
    in_dim = int(z.shape[-1])
    labels_all = [y for _, _, y in ds_train] + [y for _, _, y in ds_dev] + [y for _, _, y in ds_test]
    n_classes = int(max(labels_all)) + 1

    model = EncoderWithHead(enc, in_dim, n_classes=n_classes, amp=amp).to(device)
    params = list(base_model.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_mcc, best_state, wait = -1e9, None, 0

    for _ in tqdm(range(1, epochs + 1), desc="FullFinetune", leave=False):
        base_model.train()
        model.train()

        order = torch.randperm(len(ds_train)).tolist()
        for i0 in range(0, len(order), batch_size):
            idx = order[i0:i0 + batch_size]
            batch = [ds_train[ii] for ii in idx]
            s1 = [b[0] for b in batch]
            s2_list = [b[1] for b in batch]
            s2 = None if all(v is None for v in s2_list) else s2_list
            y = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)

            with _torch_amp.autocast("cuda", enabled=amp):
                logits = model(s1, s2, max_len=max_len)
                loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        m_dev = eval_on(model, ds_dev, max_len, batch_size, amp, device, n_classes)
        if m_dev["mcc"] > best_mcc:
            best_mcc = m_dev["mcc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    m_dev = eval_on(model, ds_dev, max_len, batch_size, amp, device, n_classes)
    m_test = eval_on(model, ds_test, max_len, batch_size, amp, device, n_classes)
    return {"dev": m_dev, "test": m_test, "select_metric": "mcc"}


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="GUE benchmark evaluation (select by DEV MCC; report MCC+macroF1).")
    ap.add_argument("--model_path", type=str, default=None, help="HF encoder path; not needed for --encoder_type=student.")
    ap.add_argument("--encoder_type", type=str, default="auto", choices=["auto", "student", "student_moe"])
    ap.add_argument("--datasets_root", type=str, required=True, help="Root dir containing GUE subset folders.")
    ap.add_argument("--datasets", type=str, default="auto", help="Comma-separated subset names, or 'auto'.")
    ap.add_argument("--mode", type=str, default="linear_probe", choices=["linear_probe", "lora", "full_finetune"])

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--local_only", action="store_true")
    ap.add_argument("--metrics_out", type=str, default=None)

    # LoRA controls
    ap.add_argument("--lora_keywords", type=str, default="", help="Comma-separated include substrings; empty -> all Linear.")
    ap.add_argument("--lora_exclude_keywords", type=str, default="", help="Comma-separated exclude substrings; empty -> default excludes.")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Student-specific
    ap.add_argument("--student_ckpt", type=str, default=None)
    ap.add_argument("--student_tokenizer", type=str, default=None)

    # Student-MoE specific (used when ckpt does not contain moe_cfg)
    ap.add_argument("--moe_num_experts", type=int, default=8)
    ap.add_argument("--moe_top_k", type=int, default=2, choices=[1, 2])
    ap.add_argument("--moe_ffn_mult", type=float, default=4.0)
    ap.add_argument("--moe_dropout", type=float, default=0.0)
    ap.add_argument("--moe_aux_weight", type=float, default=1e-2)

    args = ap.parse_args()
    set_seed(args.seed)

    is_dist, rank, world_size, local_rank = setup_distributed_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process(rank):
        print(f"[dist] enabled={is_dist} world_size={world_size} device={device}")

    root = Path(args.datasets_root)

    # Discover tasks
    if args.datasets.lower() == "auto":
        all_task_dirs = find_candidate_gue_tasks(root)
    else:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        all_task_dirs = [root / n for n in names]

    # Shard tasks across ranks
    task_dirs = [d for i, d in enumerate(all_task_dirs) if (i % world_size) == rank]
    if is_main_process(rank):
        print(f"[tasks] total={len(all_task_dirs)} | this_rank={len(task_dirs)}")

    enc = build_encoder(
        args.model_path,
        device=device,
        encoder_type=args.encoder_type,
        local_only=args.local_only,
        student_ckpt=args.student_ckpt,
        student_tokenizer=args.student_tokenizer,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_ffn_mult=args.moe_ffn_mult,
        moe_dropout=args.moe_dropout,
        moe_aux_weight=args.moe_aux_weight,
    )

    rank_results: Dict[str, Any] = {}

    for task_dir in task_dirs:
        task_name = task_dir.name
        split = load_gue_task(task_dir)
        ds_tr, ds_de, ds_te, lab2id = build_datasets_from_split(split)

        print(f"\n=== {args.mode} on: {task_name} ===")
        print(f"[schema] label_col={split.label_col} seq_cols={split.seq_cols} | train/dev/test={len(ds_tr)}/{len(ds_de)}/{len(ds_te)}")

        if args.mode == "linear_probe":
            res = run_linear_probe(
                enc, ds_tr, ds_de, ds_te,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                amp=args.amp, device=device, patience=args.patience
            )
        elif args.mode == "lora":
            include_kw = [k for k in args.lora_keywords.split(",") if k]
            exclude_kw = [k for k in args.lora_exclude_keywords.split(",") if k]
            res = run_lora(
                enc, ds_tr, ds_de, ds_te,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                amp=args.amp, device=device, patience=args.patience,
                include_kw=include_kw, exclude_kw=exclude_kw,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
            )
        else:
            res = run_full_finetune(
                enc, ds_tr, ds_de, ds_te,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                amp=args.amp, device=device, patience=args.patience
            )

        rank_results[task_name] = {
            **res,
            "label_mapping": {str(k): int(v) for k, v in lab2id.items()},
            "schema": {"label_col": split.label_col, "seq_cols": split.seq_cols},
        }

        print(f"[DEV ] MCC={res['dev']['mcc']:.4f} F1={res['dev']['f1']:.4f}")
        print(f"[TEST] MCC={res['test']['mcc']:.4f} F1={res['test']['f1']:.4f}")

    gathered = dist_all_gather_object(rank_results, world_size)

    if is_main_process(rank):
        merged: Dict[str, Any] = {}
        for part in gathered:
            merged.update(part or {})

        if args.metrics_out:
            out = Path(args.metrics_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"\n[Saved] Metrics -> {out}")

        print("\n[Done] GUE evaluation completed.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
