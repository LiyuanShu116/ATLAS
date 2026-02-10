#!/usr/bin/env python
"""
Evaluate DNA encoders (student or HF) on Genomic Benchmarks datasets with the same
protocol/style as eval_NT_benchmark.py.

Protocol (paper-style, MCC-centered):
  - Use the provided TRAIN split to build K stratified folds.
  - For each fold:
      * Train on train\fold
      * Select best checkpoint by MCC on fold (validation)
      * Evaluate the selected checkpoint on the provided TEST split
  - Report per-fold metrics and mean/std/median across folds.

Supported modes:
  - linear_probe: precompute embeddings once, train a linear classifier
  - lora: inject LoRA into encoder, train LoRA + linear head
  - full_finetune: finetune the whole encoder + linear head

Genomic Benchmarks datasets are stored as folder-of-files:
  <dataset_root>/train/<class_name>/*
  <dataset_root>/test/<class_name>/*
Optionally there may be a "valid" split. You can merge it into train via --merge_valid.

Local dataset layout (recommended for your case):
  <datasets_root>/<dataset_name>/train/<class_name>/*
  <datasets_root>/<dataset_name>/test/<class_name>/*
  [optional] <datasets_root>/<dataset_name>/valid/<class_name>/*

Or, if you put train/test directly under <datasets_root> (single dataset layout):
  <datasets_root>/train/<class_name>/*
  <datasets_root>/test/<class_name>/*

Notes:
  - Primary selection metric: MCC (robust to class imbalance).
  - Additionally report macro-F1.
"""

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp as _torch_amp
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

# Optional: Genomic Benchmarks helpers (only used for remote download / dataset info)
GB_AVAILABLE = True
try:
    from genomic_benchmarks.loc2seq import download_dataset
    from genomic_benchmarks.data_check import list_datasets as gb_list_datasets
    from genomic_benchmarks.data_check import info as gb_info
except Exception:
    GB_AVAILABLE = False
    download_dataset = None  # type: ignore
    gb_list_datasets = None  # type: ignore
    gb_info = None  # type: ignore


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset utilities (Genomic Benchmarks)
# -------------------------

def _read_seq_file(fp: Path) -> str:
    """Read a single sequence file and return an uppercase DNA string (joined across lines)."""
    txt = fp.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return ""
    return "".join(txt.split()).upper()


def _sanitize_dna(seq: str) -> str:
    """Keep A/C/G/T/N; map other letters to N."""
    allowed = set("ACGTN")
    return "".join(ch if ch in allowed else "N" for ch in seq)


def _is_folder_layout(root: Path) -> bool:
    """Return True if root looks like a folder-of-files GB dataset (train/test folders exist)."""
    return (root / "train").exists() and (root / "test").exists()


def _glob_parquet(root: Path, split: str) -> List[Path]:
    """Find parquet shards for a split under root or root/data."""
    candidates: List[Path] = []
    # Prefer <root>/data
    if (root / "data").exists():
        d = root / "data"
        # Common cases: train.parquet / test.parquet
        candidates += [p for p in [d / f"{split}.parquet"] if p.exists()]
        # Sharded cases: train-00000-of-00001-xxxx.parquet
        candidates += sorted(d.glob(f"{split}-*.parquet"))
        # Some datasets use split*.parquet
        candidates += sorted(d.glob(f"{split}*.parquet"))
    # Also support flat <root>
    candidates += [p for p in [root / f"{split}.parquet"] if p.exists()]
    candidates += sorted(root.glob(f"{split}-*.parquet"))
    candidates += sorted(root.glob(f"{split}*.parquet"))

    # Deduplicate while preserving order.
    seen = set()
    out: List[Path] = []
    for p in candidates:
        if p.is_file() and p.suffix == ".parquet" and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def _is_parquet_layout(root: Path) -> bool:
    """Return True if root looks like a parquet-based GB dataset (train/test parquet exist)."""
    tr = _glob_parquet(root, "train")
    te = _glob_parquet(root, "test")
    return (len(tr) > 0) and (len(te) > 0)


def _looks_like_gb_dataset(root: Path) -> bool:
    """Return True if root matches either folder layout or parquet layout."""
    return _is_folder_layout(root) or _is_parquet_layout(root)


class GBFolderDataset(Dataset):
    """Folder dataset holding (sequence, label_int) pairs."""

    def __init__(self, root: Path, split: str):
        self.split = split
        self.split_dir = root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        self.class_names = sorted([p.name for p in self.split_dir.iterdir() if p.is_dir()])
        if len(self.class_names) < 2:
            raise RuntimeError(f"Expected >=2 classes under {self.split_dir}, got {self.class_names}")

        self.samples: List[Tuple[Path, int]] = []
        for label, cname in enumerate(self.class_names):
            cdir = self.split_dir / cname
            for fp in sorted(cdir.rglob("*")):
                if fp.is_file():
                    self.samples.append((fp, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        fp, y = self.samples[idx]
        seq = _sanitize_dna(_read_seq_file(fp))
        return seq, y


class GBParquetDataset(Dataset):
    """Parquet dataset holding (sequence, label_int) pairs.

    Expected layout (HF-style):
      <dataset_root>/data/train.parquet
      <dataset_root>/data/test.parquet
    or sharded parquet files under <dataset_root>/data/.
    """

    def __init__(self, root: Path, split: str, label2id: Optional[Dict[Any, int]] = None):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError(
                "Parquet layout requires pandas + pyarrow/fastparquet. "
                "Please install: pip install pandas pyarrow"
            ) from e

        self.root = root
        self.split = split
        self.files = _glob_parquet(root, split)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found for split='{split}' under: {root} (or {root/'data'})")

        # Load (concatenate shards). For GB datasets the parquet files are usually small.
        dfs = [pd.read_parquet(str(fp)) for fp in self.files]
        df = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)

        if df.shape[0] == 0:
            raise RuntimeError(f"Empty parquet split='{split}' under {root}")

        # Infer columns.
        cols = list(df.columns)
        cols_l = [str(c).lower() for c in cols]

        # Sequence column heuristics.
        seq_candidates = ["sequence", "seq", "dna", "text", "input", "inputs", "x"]
        seq_col = None
        for cand in seq_candidates:
            if cand in cols_l:
                seq_col = cols[cols_l.index(cand)]
                break
        if seq_col is None:
            # Pick the first object/string-like column.
            for c in cols:
                if df[c].dtype == object:
                    seq_col = c
                    break
        if seq_col is None:
            raise RuntimeError(f"Cannot infer sequence column from parquet columns: {cols}")

        # Label column heuristics.
        label_candidates = ["label", "labels", "y", "target", "class", "cls"]
        lab_col = None
        for cand in label_candidates:
            if cand in cols_l:
                lab_col = cols[cols_l.index(cand)]
                break
        if lab_col is None:
            if len(cols) == 2:
                lab_col = cols[0] if cols[1] == seq_col else cols[1]
            else:
                # pick first non-seq numeric/object column
                for c in cols:
                    if c == seq_col:
                        continue
                    lab_col = c
                    break
        if lab_col is None:
            raise RuntimeError(f"Cannot infer label column from parquet columns: {cols}")

        seqs = df[seq_col].astype(str).tolist()
        seqs = [_sanitize_dna(s.upper()) for s in seqs]
        raw_labels = df[lab_col].tolist()

        # Build or reuse label mapping.
        if label2id is None:
            uniq = sorted(set(raw_labels), key=lambda x: str(x))
            label2id = {u: i for i, u in enumerate(uniq)}
        self.label2id = dict(label2id)

        labels = [self.label2id.get(y, None) for y in raw_labels]
        if any(v is None for v in labels):
            missing = sorted({raw_labels[i] for i, v in enumerate(labels) if v is None}, key=lambda x: str(x))
            raise RuntimeError(
                f"Found labels in split='{split}' not present in label2id. Missing: {missing[:10]}"
            )

        self.seqs = seqs
        self.labels = [int(v) for v in labels]  # type: ignore

        # For logging consistency with folder layout.
        inv = {v: k for k, v in self.label2id.items()}
        self.class_names = [str(inv[i]) for i in range(len(inv))]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.seqs[idx], self.labels[idx]


def resolve_local_dataset_root(datasets_root: Path, dataset_name: str) -> Path:
    """Resolve a dataset root for local evaluation.

    Supports:
      - dataset_name == '<name>'
      - dataset_name == '<name>/data' (will be normalized to '<name>')
    """
    ds = dataset_name.strip("/")
    parts = ds.split("/")
    if len(parts) > 1 and parts[-1] == "data":
        ds = "/".join(parts[:-1])

    cand1 = datasets_root / ds
    if cand1.exists() and _looks_like_gb_dataset(cand1):
        return cand1

    # If user passed a nested path, also try only the last folder name.
    cand2 = datasets_root / Path(ds).name
    if cand2.exists() and _looks_like_gb_dataset(cand2):
        return cand2

    # Single dataset layout under datasets_root.
    if _looks_like_gb_dataset(datasets_root):
        return datasets_root

    raise FileNotFoundError(
        f"Cannot find local dataset '{dataset_name}'. Tried:\n"
        f"  - {cand1} (expect train/test folders OR parquet under data/)\n"
        f"  - {cand2} (expect train/test folders OR parquet under data/)\n"
        f"  - {datasets_root} (expect train/test folders OR parquet under data/)\n"
        f"Your datasets_root is: {datasets_root}"
    )


def list_local_datasets(datasets_root: Path, exclude_demo: bool = False) -> List[str]:
    """List dataset names under <datasets_root> for local evaluation."""
    names: List[str] = []
    if not datasets_root.exists():
        return names
    for p in sorted(datasets_root.iterdir()):
        if not p.is_dir():
            continue
        if exclude_demo and (p.name.startswith("demo_") or p.name.startswith("dummy_")):
            continue
        if _looks_like_gb_dataset(p):
            names.append(p.name)
    return names


def build_gb_datasets(
    dataset_name: str,
    version: int,
    merge_valid: bool,
    datasets_root: Optional[str] = None,
    use_gb_download: bool = False,
) -> Tuple[Dataset, Dataset, Dict[str, int], Path]:
    """Build a dataset for evaluation (local-first)."""
    if datasets_root is not None:
        root = resolve_local_dataset_root(Path(datasets_root), dataset_name)
    else:
        if not use_gb_download:
            raise ValueError("Either --datasets_root must be set, or --use_gb_download must be enabled.")
        if not GB_AVAILABLE or download_dataset is None:
            raise ImportError(
                "genomic_benchmarks is not available, but --use_gb_download was set. "
                "Please install it: pip install genomic-benchmarks"
            )
        root = Path(download_dataset(dataset_name, version=version))

    if _is_folder_layout(root):
        ds_train = GBFolderDataset(root, split="train")
        ds_test = GBFolderDataset(root, split="test")
        lab2id = {name: i for i, name in enumerate(ds_train.class_names)}

        if merge_valid and (root / "valid").exists():
            ds_valid = GBFolderDataset(root, split="valid")
            if ds_valid.class_names != ds_train.class_names:
                raise RuntimeError(
                    f"[GB] train/valid class folders mismatch. train={ds_train.class_names} valid={ds_valid.class_names}"
                )
            ds_train = torch.utils.data.ConcatDataset([ds_train, ds_valid])  # type: ignore

        return ds_train, ds_test, lab2id, root

    if _is_parquet_layout(root):
        ds_train = GBParquetDataset(root, split="train", label2id=None)
        ds_test = GBParquetDataset(root, split="test", label2id=ds_train.label2id)
        lab2id = {name: i for i, name in enumerate(ds_train.class_names)}

        # Optional valid parquet.
        if merge_valid:
            valid_files = _glob_parquet(root, "valid")
            if valid_files:
                ds_valid = GBParquetDataset(root, split="valid", label2id=ds_train.label2id)
                ds_train = torch.utils.data.ConcatDataset([ds_train, ds_valid])  # type: ignore

        return ds_train, ds_test, lab2id, root

    raise RuntimeError(
        f"Dataset root resolved to {root}, but it is neither folder layout nor parquet layout."
    )

# -------------------------
# Stratified K-Fold (no sklearn dependency)
# -------------------------

def make_stratified_folds(labels: List[int], k: int, seed: int = 1337) -> List[Tuple[List[int], List[int]]]:
    """Return list of (train_idx, val_idx) for K folds with simple stratification by label."""
    rng = random.Random(seed)
    label_to_idx: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        label_to_idx.setdefault(int(y), []).append(i)
    for idxs in label_to_idx.values():
        rng.shuffle(idxs)

    class_slices: Dict[int, List[List[int]]] = {}
    for y, idxs in label_to_idx.items():
        n = len(idxs)
        base, rem = n // k, n % k
        slices, start = [], 0
        for f in range(k):
            size = base + (1 if f < rem else 0)
            slices.append(idxs[start:start + size])
            start += size
        class_slices[y] = slices

    folds = []
    N = len(labels)
    for f in range(k):
        val_idx = []
        for y in class_slices:
            val_idx.extend(class_slices[y][f])
        val_set = set(val_idx)
        train_idx = [i for i in range(N) if i not in val_set]
        folds.append((train_idx, val_idx))
    return folds


# -------------------------
# Metrics (MCC-centered + macro-F1)
# -------------------------

def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true == y_pred).float().mean().item()


def precision_recall_f1_macro(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int
) -> Tuple[float, float, float]:
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
    """Multi-class Matthews Correlation Coefficient (Gorodkin's generalized MCC)."""
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


def metrics_from_logits(
    y_true: torch.Tensor, logits: torch.Tensor, n_classes: int
) -> Tuple[Dict[str, float], List[List[int]]]:
    if logits.device.type == "cpu" and logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    y_pred = torch.argmax(logits, -1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1_macro(y_true, y_pred, n_classes)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    mcc = mcc_from_confusion(cm)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc}, cm

# -------------------------
# Encoders (HF + Student) -- aligned with eval_NT_benchmark.py
# -------------------------

class BaseEncoder(nn.Module):
    def encode(self, seqs: List[str], max_len: int, amp: bool = False) -> torch.Tensor:
        raise NotImplementedError


class _HFMeanPoolMixin:
    """Mean pooling with attention_mask over the last hidden states."""
    def _mean_pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            mask = torch.ones(hidden.size(0), hidden.size(1), device=hidden.device, dtype=hidden.dtype)
        else:
            mask = attention_mask.to(hidden.device, dtype=hidden.dtype)
        mask = mask.unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


class NTEncoder(BaseEncoder, _HFMeanPoolMixin):
    def __init__(self, model_path: str, device: torch.device, local_only: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=local_only)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, config=self.config, trust_remote_code=True, local_files_only=local_only).to(device)
        self.device = device

    def encode(self, seqs: List[str], max_len: int, amp: bool = False) -> torch.Tensor:
        batch = self.tokenizer(seqs, padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt", add_special_tokens=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with _torch_amp.autocast("cuda", enabled=amp):
            out = self.model(**batch, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1] if out.hidden_states is not None else out.last_hidden_state
            pooled = self._mean_pool(hidden, batch.get("attention_mask", None))
        return pooled


class HyenaEncoder(BaseEncoder, _HFMeanPoolMixin):
    def __init__(self, model_path: str, device: torch.device, local_only: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_path, config=self.config, trust_remote_code=True, local_files_only=local_only).to(device)
        self.device = device

    def encode(self, seqs: List[str], max_len: int, amp: bool = False) -> torch.Tensor:
        batch = self.tokenizer(seqs, padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt", add_special_tokens=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with _torch_amp.autocast("cuda", enabled=amp):
            out = self.model(**batch, output_hidden_states=True, return_dict=True)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else out[0]
            pooled = self._mean_pool(hidden, batch.get("attention_mask", None))
        return pooled


class GROVEREncoder(BaseEncoder, _HFMeanPoolMixin):
    def __init__(self, model_path: str, device: torch.device, local_only: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_path, config=self.config, trust_remote_code=True, local_files_only=local_only).to(device)
        self.device = device

    def encode(self, seqs: List[str], max_len: int, amp: bool = False) -> torch.Tensor:
        batch = self.tokenizer(seqs, padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt", add_special_tokens=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with _torch_amp.autocast("cuda", enabled=amp):
            out = self.model(**batch, output_hidden_states=True, return_dict=True)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else out[0]
            pooled = self._mean_pool(hidden, batch.get("attention_mask", None))
        return pooled


class DNABERT2Encoder(BaseEncoder, _HFMeanPoolMixin):
    """DNABERT-2 wrapper (e.g., local path: src/hf/DNABERT-2-117M)."""
    def __init__(self, model_path: str, device: torch.device, local_only: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_path, config=self.config, trust_remote_code=True, local_files_only=local_only).to(device)
        self.device = device

    def encode(self, seqs: List[str], max_len: int, amp: bool = False) -> torch.Tensor:
        batch = self.tokenizer(seqs, padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt", add_special_tokens=True)
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
    Wrapper for StudentMamba2 checkpoints (dense or MoE); returns pooled embedding per sequence.

    Robust checkpoint loading:
      - Supports raw state_dict, {"model": ...}, {"state_dict": ...}
      - Strips common DDP/module prefixes (e.g., "module.")
      - Strips wrapper prefixes (e.g., "backbone.") that often appear in stage-2 checkpoints
    """
    def __init__(
        self,
        ckpt_path: str,
        tokenizer_path: str,
        device: torch.device,
        local_only: bool = False,
        force_moe: bool = False,
        random_init: bool = False,
        # MoE knobs (used only when checkpoint does not carry moe_cfg)
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_ffn_mult: float = 4.0,
        moe_dropout: float = 0.0,
        moe_aux_weight: float = 1e-2,
        verbose: bool = True,
    ):
        super().__init__()
        self.is_student = True
        self.device = device

        # Robust import across possible project layouts.
        StudentMamba2 = None
        tried = []
        for mod in [
            "src.student_moe", "src.models.student_moe", "student_moe", "models.student_moe",
            "src.student", "src.models.student", "student", "models.student",
        ]:
            try:
                m = __import__(mod, fromlist=["StudentMamba2"])
                StudentMamba2 = getattr(m, "StudentMamba2")
                break
            except Exception as e:
                tried.append(f"{mod}: {type(e).__name__}")
                continue
        if StudentMamba2 is None:
            raise ImportError("Cannot import StudentMamba2. Tried:\n  - " + "\n  - ".join(tried))

        # Tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=local_only
        )
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        ckpt = torch.load(ckpt_path, map_location="cpu")

        def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
            if isinstance(obj, dict):
                for key in ("state_dict", "model", "student_state_dict", "net", "weights"):
                    if key in obj and isinstance(obj[key], dict):
                        return obj[key]
                # Raw state_dict.
                if all(isinstance(k, str) for k in obj.keys()) and any(isinstance(v, torch.Tensor) for v in obj.values()):
                    return obj  # type: ignore
            raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

        def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            if not sd:
                return sd
            if any(k.startswith("module.") for k in sd.keys()):
                return {k.replace("module.", "", 1): v for k, v in sd.items()}
            return sd

        def _maybe_strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            if not sd:
                return sd
            keys = list(sd.keys())
            n_pref = sum(1 for k in keys if k.startswith(prefix))
            if n_pref >= int(0.6 * len(keys)):
                return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            return sd

        state = _strip_module_prefix(_extract_state_dict(ckpt))

        # Strip common wrapper prefixes (apply twice to handle nested prefix).
        for pref in ("backbone.", "encoder.", "student.", "model.", "net."):
            state = _maybe_strip_prefix(state, pref)
            state = _maybe_strip_prefix(state, pref)

        cfg: Dict[str, Any] = {}
        if isinstance(ckpt, dict):
            cfg = ckpt.get("dense_cfg") or ckpt.get("config") or ckpt.get("student_core_config") or ckpt.get("backbone_cfg") or {}

        moe_cfg: Dict[str, Any] = {}
        if isinstance(ckpt, dict) and isinstance(ckpt.get("moe_cfg", None), dict):
            moe_cfg = ckpt["moe_cfg"]
        elif isinstance(cfg.get("moe_cfg", None), dict):
            moe_cfg = cfg["moe_cfg"]

        # Core hyper-params (fallbacks match training defaults).
        d_model = int(cfg.get("d_model", 384))
        n_layers = int(cfg.get("n_layers", 24))
        d_state = int(cfg.get("d_state", 16))
        expand = int(cfg.get("expand", 2))
        d_conv = int(cfg.get("d_conv", cfg.get("dconv", 4)))

        # Infer which blocks use MoE if moe_cfg missing.
        K = max(0, n_layers // 4)
        M = n_layers - K
        moe_block_ids = set()
        for k in state.keys():
            if (".moe." in k) or (".router." in k) or (".experts." in k) or ("router.proj" in k):
                m = re.match(r"^blocks\.(\d+)\.", k)
                if m is not None:
                    moe_block_ids.add(int(m.group(1)))

        use_moe_mamba = moe_cfg.get("use_moe_mamba", None)
        use_moe_attn = moe_cfg.get("use_moe_attn", None)

        if use_moe_mamba is None and use_moe_attn is None:
            use_moe_mamba = any(i < M for i in moe_block_ids)
            use_moe_attn = any(i >= M for i in moe_block_ids)

        use_moe = bool(use_moe_mamba or use_moe_attn or force_moe)

        if moe_cfg:
            moe_num_experts = int(moe_cfg.get("moe_num_experts", moe_cfg.get("num_experts", moe_num_experts)))
            moe_top_k = int(moe_cfg.get("moe_top_k", moe_cfg.get("top_k", moe_top_k)))
            moe_ffn_mult = float(moe_cfg.get("moe_ffn_mult", moe_cfg.get("ffn_mult", moe_ffn_mult)))
            moe_dropout = float(moe_cfg.get("moe_dropout", moe_cfg.get("dropout", moe_dropout)))
            moe_aux_weight = float(moe_cfg.get("moe_aux_weight", moe_cfg.get("aux_weight", moe_aux_weight)))

        # Instantiate with signature-aware kwargs.
        import inspect
        init_sig = inspect.signature(StudentMamba2.__init__)
        kwargs = dict(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            pad_id=self.pad_id,
            # MoE args (used only if StudentMamba2 supports them).
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

        kwargs = {k: v for k, v in kwargs.items() if k in init_sig.parameters}
        self.model = StudentMamba2(**kwargs)

        if not random_init:
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if verbose:
                if unexpected:
                    print(f"[student{'+moe' if use_moe else ''}][warn] Unexpected keys while loading: {len(unexpected)}")
                    print(f"[student{'+moe' if use_moe else ''}][warn] Unexpected sample: {unexpected[:5]}")
                if missing:
                    print(f"[student{'+moe' if use_moe else ''}][warn] Missing keys while loading: {len(missing)}")
                    print(f"[student{'+moe' if use_moe else ''}][warn] Missing sample: {missing[:5]}")
        else:
            if verbose:
                print(f"[student{'+moe' if use_moe else ''}][info] Random init enabled: skip loading checkpoint weights.")

        self.model.to(device)
        self.model.eval()

    def encode(self, seqs: List[str], max_len: int, amp: bool = False) -> torch.Tensor:
        batch = self.tokenizer(
            seqs, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt", add_special_tokens=True
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Pass compute_logits=False if supported (saves memory/compute).
        import inspect
        fwd_sig = inspect.signature(self.model.forward)
        fwd_kwargs = dict(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask", None))
        if "compute_logits" in fwd_sig.parameters:
            fwd_kwargs["compute_logits"] = False

        with _torch_amp.autocast("cuda", enabled=amp):
            out = self.model(**fwd_kwargs)
            pooled = out["pooled"]
        return pooled


def build_encoder(
    model_path: Optional[str],
    device: torch.device,
    encoder_type: str = "auto",
    local_only: bool = False,
    student_ckpt: Optional[str] = None,
    student_tokenizer: Optional[str] = None,
    student_random_init: bool = False,
    # MoE knobs (used only for student/student_moe if ckpt does not provide moe_cfg)
    moe_num_experts: int = 8,
    moe_top_k: int = 2,
    moe_ffn_mult: float = 4.0,
    moe_dropout: float = 0.0,
    moe_aux_weight: float = 1e-2,
) -> BaseEncoder:
    """Build an encoder from a HF path or a student checkpoint."""
    if encoder_type in ("student", "student_moe"):
        if not student_ckpt or not student_tokenizer:
            raise ValueError("--encoder_type student/student_moe requires --student_ckpt and --student_tokenizer")
        force_moe = (encoder_type == "student_moe")
        return StudentEncoder(
            ckpt_path=student_ckpt,
            tokenizer_path=student_tokenizer,
            device=device,
            local_only=local_only,
            force_moe=force_moe,
            random_init=student_random_init,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_ffn_mult=moe_ffn_mult,
            moe_dropout=moe_dropout,
            moe_aux_weight=moe_aux_weight,
            verbose=True,
        )

    if not model_path:
        raise ValueError("--model_path is required unless --encoder_type=student")

    mt = ""
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_only)
        mt = (getattr(cfg, "model_type", "") or "").lower()
    except Exception:
        pass
    path_l = (model_path or "").lower()

    if encoder_type == "dnabert2" or ("dnabert" in mt) or ("dnabert" in path_l):
        return DNABERT2Encoder(model_path, device, local_only)

    if encoder_type == "nt" or ("esm" in mt or "nucleotide" in mt):
        return NTEncoder(model_path, device, local_only)
    if encoder_type == "hyena" or "hyena" in mt:
        return HyenaEncoder(model_path, device, local_only)
    if encoder_type == "grover" or "grover" in mt:
        return GROVEREncoder(model_path, device, local_only)

    # Auto fallback.
    for enc_cls in (DNABERT2Encoder, NTEncoder, HyenaEncoder, GROVEREncoder):
        try:
            return enc_cls(model_path, device, local_only)
        except Exception:
            continue
    return GROVEREncoder(model_path, device, local_only)


# -------------------------
# LoRA wrappers (aligned with eval_NT_benchmark.py)
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
    verbose: bool = True
) -> int:
    """Replace selected nn.Linear modules by LoRALinear in-place."""
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
    """Re-initialize LoRA adapters so each CV fold starts from the same init."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.kaiming_uniform_(m.A, a=math.sqrt(5))
            nn.init.zeros_(m.B)


# -------------------------
# Simple heads
# -------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class EncoderWithHead(nn.Module):
    def __init__(self, enc: BaseEncoder, in_dim: int, n_classes: int, amp: bool = False):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(in_dim, n_classes)
        self.amp = amp

    def forward(self, seqs: List[str], max_len: int) -> torch.Tensor:
        with _torch_amp.autocast("cuda", enabled=self.amp):
            z = self.enc.encode(seqs, max_len=max_len, amp=self.amp)
            return self.head(z)


# -------------------------
# Helpers
# -------------------------

def embed_with_pbar(enc: BaseEncoder, ds: Dataset, batch_size: int, max_len: int, amp: bool, desc: str):
    feats, labs = [], []
    pbar = tqdm(range(0, len(ds), batch_size), desc=desc, leave=False)
    for i in pbar:
        seqs = [ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]
        labels = [ds[j][1] for j in range(i, min(i + batch_size, len(ds)))]
        z = enc.encode(seqs, max_len=max_len, amp=amp)
        feats.append(z.cpu())
        labs.extend(labels)
    return torch.cat(feats, 0), torch.tensor(labs, dtype=torch.long)


def agg_stats(values: List[float]) -> Tuple[float, float, float]:
    t = torch.tensor(values, dtype=torch.float32)
    mean = float(t.mean().item())
    std = float(t.std(unbiased=True).item() if len(t) > 1 else 0.0)
    med = float(t.median().item())
    return mean, std, med


# -------------------------
# Linear Probe CV (train folds, evaluate on fixed test)
# -------------------------

def run_linear_probe_cv(
    enc: BaseEncoder,
    ds_train: Dataset,
    ds_test: Dataset,
    k_folds: int,
    cv_seed: int,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    device: torch.device,
):
    labels_train = [ds_train[i][1] for i in range(len(ds_train))]
    labels_test = [ds_test[i][1] for i in range(len(ds_test))]
    n_classes = max(labels_train + labels_test) + 1

    # Precompute embeddings once.
    Xtr_all, ytr_all = embed_with_pbar(enc, ds_train, batch_size, max_len, amp, desc="Embed train")
    Xte, yte = embed_with_pbar(enc, ds_test, batch_size, max_len, amp, desc="Embed test")

    folds = make_stratified_folds(ytr_all.tolist(), k=k_folds, seed=cv_seed)
    val_mcc_list, test_mcc_list = [], []
    val_f1_list, test_f1_list = [], []
    val_acc_list, test_acc_list = [], []

    for f, (tr_idx, va_idx) in enumerate(folds, 1):
        clf = LinearHead(Xtr_all.size(1), n_classes).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
        Xva, yva = Xtr_all[va_idx], ytr_all[va_idx]

        best_mcc, best_state, patience, wait = -1e9, None, 3, 0
        for _ in tqdm(range(1, epochs + 1), desc=f"LinearProbe fold {f}/{k_folds}", leave=False):
            clf.train()
            order = torch.randperm(Xtr.size(0))
            for i in range(0, Xtr.size(0), batch_size):
                idx = order[i:i + batch_size]
                xb = Xtr[idx].to(device)
                yb = ytr[idx].to(device)
                logits = clf(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
                opt.step()

            # Validation (MCC) for checkpoint selection.
            clf.eval()
            logits_all = []
            with torch.no_grad():
                for i in range(0, Xva.size(0), batch_size):
                    xb = Xva[i:i + batch_size].to(device)
                    logits_all.append(clf(xb).cpu())
            logits = torch.cat(logits_all, 0)
            m_val, _ = metrics_from_logits(yva, logits, n_classes)

            if m_val["mcc"] > best_mcc:
                best_mcc = m_val["mcc"]
                best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            clf.load_state_dict(best_state)

        clf.eval()
        with torch.no_grad():
            # Final val metrics.
            val_logits = []
            for i in range(0, Xva.size(0), batch_size):
                val_logits.append(clf(Xva[i:i + batch_size].to(device)).cpu())
            m_val, _ = metrics_from_logits(yva, torch.cat(val_logits, 0), n_classes)

            # Test metrics.
            test_logits = []
            for i in range(0, Xte.size(0), batch_size):
                test_logits.append(clf(Xte[i:i + batch_size].to(device)).cpu())
            m_test, _ = metrics_from_logits(yte, torch.cat(test_logits, 0), n_classes)

        val_mcc_list.append(m_val["mcc"])
        test_mcc_list.append(m_test["mcc"])
        val_f1_list.append(m_val["f1"])
        test_f1_list.append(m_test["f1"])
        val_acc_list.append(m_val["acc"])
        test_acc_list.append(m_test["acc"])

        mean_t_mcc, std_t_mcc, _ = agg_stats(test_mcc_list)
        mean_t_f1, std_t_f1, _ = agg_stats(test_f1_list)
        mean_t_acc, std_t_acc, _ = agg_stats(test_acc_list)
        print(
            f"[CV][fold {f}/{k_folds}] "
            f"val-MCC={m_val['mcc']:.4f} val-F1={m_val['f1']:.4f} | "
            f"test-MCC={m_test['mcc']:.4f} test-F1={m_test['f1']:.4f} | "
            f"running test-MCC mean={mean_t_mcc:.4f} std={std_t_mcc:.4f} ; "
            f"test-F1 mean={mean_t_f1:.4f} std={std_t_f1:.4f}"
        )
        print(
            f"[CV-ACC][fold {f}/{k_folds}] "
            f"val-ACC={m_val['acc']:.4f} | "
            f"test-ACC={m_test['acc']:.4f} | "
            f"running test-ACC mean={mean_t_acc:.4f} std={std_t_acc:.4f}"
        )

    v_mcc_mean, v_mcc_std, v_mcc_med = agg_stats(val_mcc_list)
    t_mcc_mean, t_mcc_std, t_mcc_med = agg_stats(test_mcc_list)
    v_f1_mean, v_f1_std, v_f1_med = agg_stats(val_f1_list)
    t_f1_mean, t_f1_std, t_f1_med = agg_stats(test_f1_list)
    v_acc_mean, v_acc_std, v_acc_med = agg_stats(val_acc_list)
    t_acc_mean, t_acc_std, t_acc_med = agg_stats(test_acc_list)

    return {
        "cv_val_mcc": {"per_fold": val_mcc_list, "mean": v_mcc_mean, "std": v_mcc_std, "median": v_mcc_med},
        "cv_test_mcc": {"per_fold": test_mcc_list, "mean": t_mcc_mean, "std": t_mcc_std, "median": t_mcc_med},
        "cv_val_f1": {"per_fold": val_f1_list, "mean": v_f1_mean, "std": v_f1_std, "median": v_f1_med},
        "cv_test_f1": {"per_fold": test_f1_list, "mean": t_f1_mean, "std": t_f1_std, "median": t_f1_med},
        "cv_val_acc": {"per_fold": val_acc_list, "mean": v_acc_mean, "std": v_acc_std, "median": v_acc_med},
        "cv_test_acc": {"per_fold": test_acc_list, "mean": t_acc_mean, "std": t_acc_std, "median": t_acc_med},
    }


# -------------------------
# LoRA CV
# -------------------------

def run_lora_cv(
    enc: BaseEncoder,
    ds_train: Dataset,
    ds_test: Dataset,
    k_folds: int,
    cv_seed: int,
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
):
    base_model = getattr(enc, "model", None)
    if base_model is None:
        raise RuntimeError("Encoder must expose .model for LoRA injection.")

    # Freeze base weights (LoRA keeps its own trainable params).
    for p in base_model.parameters():
        p.requires_grad_(False)

    # Default include/exclude heuristics.
    is_student = bool(getattr(enc, "is_student", False))
    model_type = (getattr(getattr(enc, "config", None), "model_type", "") or "").lower()

    if not include_kw:
        if is_student:
            include_kw = []
        elif ("hyena" in model_type) or ("grover" in model_type):
            include_kw = ["mlp", "proj", "fc", "head", "classifier", "out"]
        else:
            include_kw = ["query", "key", "value", "output.dense", "intermediate.dense"]

    if not exclude_kw:
        exclude_kw = ["embedding", "embeddings", "norm", "layernorm"]

    inject_lora(base_model, include_kw, exclude_kw, r=r, alpha=alpha, dropout=dropout, verbose=True)
    base_model.to(device)

    # Discover feature dim.
    with torch.no_grad():
        base_model.eval()
        z = enc.encode(["ACGT"], max_len=max_len, amp=amp)
    in_dim = int(z.shape[-1])

    labels_all = [ds_train[i][1] for i in range(len(ds_train))]
    labels_test = [ds_test[i][1] for i in range(len(ds_test))]
    n_classes = max(labels_all + labels_test) + 1
    folds = make_stratified_folds(labels_all, k=k_folds, seed=cv_seed)

    val_mcc_list, test_mcc_list = [], []
    val_f1_list, test_f1_list = [], []
    val_acc_list, test_acc_list = [], []

    def iterate(ds: Dataset):
        for i in range(0, len(ds), batch_size):
            seqs = [ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]
            labels = torch.tensor([ds[j][1] for j in range(i, min(i + batch_size, len(ds)))],
                                  dtype=torch.long, device=device)
            yield seqs, labels

    @torch.no_grad()
    def eval_on(model: nn.Module, ds: Dataset) -> Dict[str, float]:
        base_model.eval()
        model.eval()
        logits_all, y_all = [], []
        for seqs, labels in iterate(ds):
            with _torch_amp.autocast("cuda", enabled=amp):
                logits = model(seqs, max_len=max_len)
            logits_all.append(logits.cpu())
            y_all.extend(labels.cpu().tolist())
        logits_all = torch.cat(logits_all, 0)
        y_true = torch.tensor(y_all, dtype=torch.long)
        m, _ = metrics_from_logits(y_true, logits_all, n_classes)
        return m

    for f, (tr_idx, va_idx) in enumerate(folds, 1):
        ds_tr = Subset(ds_train, tr_idx)
        ds_va = Subset(ds_train, va_idx)

        reset_lora_params(base_model)
        model = EncoderWithHead(enc, in_dim, n_classes=n_classes, amp=amp).to(device)

        # Trainable params = LoRA params + head params.
        trainable = [p for p in base_model.parameters() if p.requires_grad] + list(model.head.parameters())
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_mcc, best_state, patience, wait = -1e9, None, 3, 0
        for _ in tqdm(range(1, epochs + 1), desc=f"LoRA fold {f}/{k_folds}", leave=False):
            base_model.train()
            model.train()

            order = torch.randperm(len(ds_tr)).tolist()
            for i0 in range(0, len(order), batch_size):
                idx = order[i0:i0 + batch_size]
                seqs = [ds_tr[ii][0] for ii in idx]
                labels = torch.tensor([ds_tr[ii][1] for ii in idx], dtype=torch.long, device=device)

                with _torch_amp.autocast("cuda", enabled=amp):
                    logits = model(seqs, max_len=max_len)
                    loss = loss_fn(logits, labels)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()

            m_val = eval_on(model, ds_va)
            if m_val["mcc"] > best_mcc:
                best_mcc = m_val["mcc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        m_val = eval_on(model, ds_va)
        m_test = eval_on(model, ds_test)

        val_mcc_list.append(m_val["mcc"])
        test_mcc_list.append(m_test["mcc"])
        val_f1_list.append(m_val["f1"])
        test_f1_list.append(m_test["f1"])
        val_acc_list.append(m_val["acc"])
        test_acc_list.append(m_test["acc"])

        mean_t_mcc, std_t_mcc, _ = agg_stats(test_mcc_list)
        mean_t_f1, std_t_f1, _ = agg_stats(test_f1_list)
        mean_t_acc, std_t_acc, _ = agg_stats(test_acc_list)
        print(
            f"[CV][fold {f}/{k_folds}] "
            f"val-MCC={m_val['mcc']:.4f} val-F1={m_val['f1']:.4f} | "
            f"test-MCC={m_test['mcc']:.4f} test-F1={m_test['f1']:.4f} | "
            f"running test-MCC mean={mean_t_mcc:.4f} std={std_t_mcc:.4f} ; "
            f"test-F1 mean={mean_t_f1:.4f} std={std_t_f1:.4f}"
        )
        print(
            f"[CV-ACC][fold {f}/{k_folds}] "
            f"val-ACC={m_val['acc']:.4f} | "
            f"test-ACC={m_test['acc']:.4f} | "
            f"running test-ACC mean={mean_t_acc:.4f} std={std_t_acc:.4f}"
        )

    v_mcc_mean, v_mcc_std, v_mcc_med = agg_stats(val_mcc_list)
    t_mcc_mean, t_mcc_std, t_mcc_med = agg_stats(test_mcc_list)
    v_f1_mean, v_f1_std, v_f1_med = agg_stats(val_f1_list)
    t_f1_mean, t_f1_std, t_f1_med = agg_stats(test_f1_list)
    v_acc_mean, v_acc_std, v_acc_med = agg_stats(val_acc_list)
    t_acc_mean, t_acc_std, t_acc_med = agg_stats(test_acc_list)

    return {
        "cv_val_mcc": {"per_fold": val_mcc_list, "mean": v_mcc_mean, "std": v_mcc_std, "median": v_mcc_med},
        "cv_test_mcc": {"per_fold": test_mcc_list, "mean": t_mcc_mean, "std": t_mcc_std, "median": t_mcc_med},
        "cv_val_f1": {"per_fold": val_f1_list, "mean": v_f1_mean, "std": v_f1_std, "median": v_f1_med},
        "cv_test_f1": {"per_fold": test_f1_list, "mean": t_f1_mean, "std": t_f1_std, "median": t_f1_med},
        "cv_val_acc": {"per_fold": val_acc_list, "mean": v_acc_mean, "std": v_acc_std, "median": v_acc_med},
        "cv_test_acc": {"per_fold": test_acc_list, "mean": t_acc_mean, "std": t_acc_std, "median": t_acc_med},
    }


# -------------------------
# Full finetune CV
# -------------------------

def run_full_finetune_cv(
    enc: BaseEncoder,
    ds_train: Dataset,
    ds_test: Dataset,
    k_folds: int,
    cv_seed: int,
    max_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    device: torch.device,
):
    base_model = getattr(enc, "model", None)
    if base_model is None:
        raise RuntimeError("Encoder must expose .model for full finetune.")

    for p in base_model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        base_model.eval()
        z = enc.encode(["ACGT"], max_len=max_len, amp=amp)
    in_dim = int(z.shape[-1])

    labels_all = [ds_train[i][1] for i in range(len(ds_train))]
    labels_test = [ds_test[i][1] for i in range(len(ds_test))]
    n_classes = max(labels_all + labels_test) + 1
    folds = make_stratified_folds(labels_all, k=k_folds, seed=cv_seed)

    init_state = copy.deepcopy(base_model.state_dict())

    val_mcc_list, test_mcc_list = [], []
    val_f1_list, test_f1_list = [], []
    val_acc_list, test_acc_list = [], []

    def iterate(ds: Dataset):
        for i in range(0, len(ds), batch_size):
            seqs = [ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]
            labels = torch.tensor([ds[j][1] for j in range(i, min(i + batch_size, len(ds)))],
                                  dtype=torch.long, device=device)
            yield seqs, labels

    @torch.no_grad()
    def eval_on(model: nn.Module, ds: Dataset) -> Dict[str, float]:
        base_model.eval()
        model.eval()
        logits_all, y_all = [], []
        for seqs, labels in iterate(ds):
            with _torch_amp.autocast("cuda", enabled=amp):
                logits = model(seqs, max_len=max_len)
            logits_all.append(logits.cpu())
            y_all.extend(labels.cpu().tolist())
        logits_all = torch.cat(logits_all, 0)
        y_true = torch.tensor(y_all, dtype=torch.long)
        m, _ = metrics_from_logits(y_true, logits_all, n_classes)
        return m

    for f, (tr_idx, va_idx) in enumerate(folds, 1):
        base_model.load_state_dict(init_state, strict=True)
        model = EncoderWithHead(enc, in_dim, n_classes=n_classes, amp=amp).to(device)
        params = list(base_model.parameters()) + list(model.head.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        ds_tr = Subset(ds_train, tr_idx)
        ds_va = Subset(ds_train, va_idx)

        best_mcc, best_state, patience, wait = -1e9, None, 5, 0
        for _ in tqdm(range(1, epochs + 1), desc=f"FullFinetune fold {f}/{k_folds}", leave=False):
            base_model.train()
            model.train()

            order = torch.randperm(len(ds_tr)).tolist()
            for i0 in range(0, len(order), batch_size):
                idx = order[i0:i0 + batch_size]
                seqs = [ds_tr[ii][0] for ii in idx]
                labels = torch.tensor([ds_tr[ii][1] for ii in idx], dtype=torch.long, device=device)

                with _torch_amp.autocast("cuda", enabled=amp):
                    logits = model(seqs, max_len=max_len)
                    loss = loss_fn(logits, labels)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            m_val = eval_on(model, ds_va)
            if m_val["mcc"] > best_mcc:
                best_mcc = m_val["mcc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        m_val = eval_on(model, ds_va)
        m_test = eval_on(model, ds_test)

        val_mcc_list.append(m_val["mcc"])
        test_mcc_list.append(m_test["mcc"])
        val_f1_list.append(m_val["f1"])
        test_f1_list.append(m_test["f1"])
        val_acc_list.append(m_val["acc"])
        test_acc_list.append(m_test["acc"])

        mean_t_mcc, std_t_mcc, _ = agg_stats(test_mcc_list)
        mean_t_f1, std_t_f1, _ = agg_stats(test_f1_list)
        mean_t_acc, std_t_acc, _ = agg_stats(test_acc_list)
        print(
            f"[CV][fold {f}/{k_folds}] "
            f"val-MCC={m_val['mcc']:.4f} val-F1={m_val['f1']:.4f} | "
            f"test-MCC={m_test['mcc']:.4f} test-F1={m_test['f1']:.4f} | "
            f"running test-MCC mean={mean_t_mcc:.4f} std={std_t_mcc:.4f} ; "
            f"test-F1 mean={mean_t_f1:.4f} std={std_t_f1:.4f}"
        )
        print(
            f"[CV-ACC][fold {f}/{k_folds}] "
            f"val-ACC={m_val['acc']:.4f} | "
            f"test-ACC={m_test['acc']:.4f} | "
            f"running test-ACC mean={mean_t_acc:.4f} std={std_t_acc:.4f}"
        )

    v_mcc_mean, v_mcc_std, v_mcc_med = agg_stats(val_mcc_list)
    t_mcc_mean, t_mcc_std, t_mcc_med = agg_stats(test_mcc_list)
    v_f1_mean, v_f1_std, v_f1_med = agg_stats(val_f1_list)
    t_f1_mean, t_f1_std, t_f1_med = agg_stats(test_f1_list)
    v_acc_mean, v_acc_std, v_acc_med = agg_stats(val_acc_list)
    t_acc_mean, t_acc_std, t_acc_med = agg_stats(test_acc_list)

    return {
        "cv_val_mcc": {"per_fold": val_mcc_list, "mean": v_mcc_mean, "std": v_mcc_std, "median": v_mcc_med},
        "cv_test_mcc": {"per_fold": test_mcc_list, "mean": t_mcc_mean, "std": t_mcc_std, "median": t_mcc_med},
        "cv_val_f1": {"per_fold": val_f1_list, "mean": v_f1_mean, "std": v_f1_std, "median": v_f1_med},
        "cv_test_f1": {"per_fold": test_f1_list, "mean": t_f1_mean, "std": t_f1_std, "median": t_f1_med},
        "cv_val_acc": {"per_fold": val_acc_list, "mean": v_acc_mean, "std": v_acc_std, "median": v_acc_med},
        "cv_test_acc": {"per_fold": test_acc_list, "mean": t_acc_mean, "std": t_acc_std, "median": t_acc_med},
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Genomic Benchmarks evaluation (MCC + macro-F1), NT-style CV protocol")
    ap.add_argument("--model_path", type=str, default=None, help="HF encoder path; not needed for --encoder_type=student.")
    ap.add_argument("--encoder_type", type=str, default="auto",
                    choices=["auto", "nt", "hyena", "grover", "student", "student_moe", "dnabert2"])

    ap.add_argument("--datasets", type=str, required=True,
                    help="Comma-separated GB dataset names; or 'auto' to evaluate all (optionally --exclude_demo).")
    ap.add_argument("--version", type=int, default=0, help="GB dataset version (usually 0).")
    ap.add_argument("--exclude_demo", action="store_true", help="When --datasets=auto, exclude demo_* and dummy_* datasets.")
    ap.add_argument("--merge_valid", action="store_true", help="If 'valid' split exists, merge it into train.")
    ap.add_argument("--datasets_root", type=str, default="data/benchmark/genomic_benchmark",
                    help="Local root folder containing downloaded Genomic Benchmarks datasets. "
                         "Expected: <datasets_root>/<dataset_name>/{train,test,valid}/...")
    ap.add_argument("--use_gb_download", action="store_true",
                    help="If set and local dataset not used, download datasets via genomic_benchmarks (requires package).")

    ap.add_argument("--mode", type=str, default="linear_probe", choices=["linear_probe", "lora", "full_finetune"])
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--metrics_out", type=str, default=None)
    ap.add_argument("--local_only", action="store_true")

    # K-fold controls (folds built on TRAIN only).
    ap.add_argument("--k_folds", type=int, default=10)
    ap.add_argument("--cv_seed", type=int, default=1337)

    # LoRA controls.
    ap.add_argument("--lora_keywords", type=str, default="", help="Comma-separated include substrings; empty -> auto")
    ap.add_argument("--lora_exclude_keywords", type=str, default="", help="Comma-separated exclude substrings; empty -> auto")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Student-specific.
    ap.add_argument("--student_ckpt", type=str, default=None)
    ap.add_argument("--student_tokenizer", type=str, default=None)
    ap.add_argument("--student_random_init", action="store_true",
                    help="Build the same student architecture but DO NOT load checkpoint weights (random init baseline).")

    # Student-MoE specific (used when ckpt does not contain moe_cfg).
    ap.add_argument("--moe_num_experts", type=int, default=8)
    ap.add_argument("--moe_top_k", type=int, default=2, choices=[1, 2])
    ap.add_argument("--moe_ffn_mult", type=float, default=4.0)
    ap.add_argument("--moe_dropout", type=float, default=0.0)
    ap.add_argument("--moe_aux_weight", type=float, default=1e-2)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Resolve dataset list.
    if args.datasets.lower() == "auto":
        # Prefer local listing when datasets_root exists.
        ds_names = []
        if args.datasets_root and Path(args.datasets_root).exists():
            ds_names = list_local_datasets(Path(args.datasets_root), exclude_demo=args.exclude_demo)
        if not ds_names:
            if GB_AVAILABLE and gb_list_datasets is not None:
                ds_names = gb_list_datasets()
                if args.exclude_demo:
                    ds_names = [n for n in ds_names if not (n.startswith("demo_") or n.startswith("dummy_"))]
            else:
                raise RuntimeError(
                    "No local datasets found under --datasets_root, and genomic_benchmarks is unavailable. "
                    "Please set --datasets to a specific dataset name, or install genomic-benchmarks."
                )
    else:
        ds_names = [s.strip() for s in args.datasets.split(",") if s.strip()]

    enc = build_encoder(
        args.model_path,
        device=device,
        encoder_type=args.encoder_type,
        local_only=args.local_only,
        student_ckpt=args.student_ckpt,
        student_tokenizer=args.student_tokenizer,
        student_random_init=args.student_random_init,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_ffn_mult=args.moe_ffn_mult,
        moe_dropout=args.moe_dropout,
        moe_aux_weight=args.moe_aux_weight,
    )

    all_results: Dict[str, Any] = {}
    for ds_name in ds_names:
        print("\n" + "=" * 90)
        print(f"[GB] Dataset: {ds_name} (version={args.version})")
        if GB_AVAILABLE and gb_info is not None and args.use_gb_download and (not args.datasets_root):
            try:
                print(gb_info(ds_name, version=args.version))
            except Exception as e:
                print(f"[GB][warn] Cannot print dataset info: {type(e).__name__}: {e}")

        ds_tr, ds_te, lab2id, ds_root = build_gb_datasets(
            ds_name,
            version=args.version,
            merge_valid=args.merge_valid,
            datasets_root=args.datasets_root,
            use_gb_download=args.use_gb_download,
        )
        n_classes = len(lab2id)
        print(f"[data] root={ds_root}")
        print(f"[data] train={len(ds_tr)} test={len(ds_te)} n_classes={n_classes} classes={list(lab2id.keys())}")
        print(f"=== {args.mode} | K={args.k_folds} (select by MCC; also report macro-F1) ===")

        if args.mode == "linear_probe":
            res = run_linear_probe_cv(
                enc, ds_tr, ds_te,
                k_folds=args.k_folds, cv_seed=args.cv_seed,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                amp=args.amp, device=device,
            )
        elif args.mode == "lora":
            include_kw = [k for k in args.lora_keywords.split(",") if k]
            exclude_kw = [k for k in args.lora_exclude_keywords.split(",") if k]
            res = run_lora_cv(
                enc, ds_tr, ds_te,
                k_folds=args.k_folds, cv_seed=args.cv_seed,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                amp=args.amp, device=device,
                include_kw=include_kw, exclude_kw=exclude_kw,
                r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            )
        else:
            res = run_full_finetune_cv(
                enc, ds_tr, ds_te,
                k_folds=args.k_folds, cv_seed=args.cv_seed,
                max_len=args.max_len, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                amp=args.amp, device=device,
            )

        res["encoder_info"] = {
            "encoder_type": args.encoder_type,
            "mode": args.mode,
            "student_random_init": bool(args.student_random_init),
            "max_len": int(args.max_len),
        }
        res["label_mapping"] = {str(k): int(v) for k, v in lab2id.items()}
        all_results[ds_name] = res

        # Print summaries.
        vm, vs, vmed = res["cv_val_mcc"]["mean"], res["cv_val_mcc"]["std"], res["cv_val_mcc"]["median"]
        tm, ts, tmed = res["cv_test_mcc"]["mean"], res["cv_test_mcc"]["std"], res["cv_test_mcc"]["median"]
        print(f"[SUMMARY-MCC] {ds_name} | VAL mean={vm:.4f} std={vs:.4f} median={vmed:.4f}")
        print(f"[SUMMARY-MCC] {ds_name} | TEST mean={tm:.4f} std={ts:.4f} median={tmed:.4f}")

        vf1m, vf1s, vf1med = res["cv_val_f1"]["mean"], res["cv_val_f1"]["std"], res["cv_val_f1"]["median"]
        tf1m, tf1s, tf1med = res["cv_test_f1"]["mean"], res["cv_test_f1"]["std"], res["cv_test_f1"]["median"]
        print(f"[SUMMARY-F1 ] {ds_name} | VAL mean={vf1m:.4f} std={vf1s:.4f} median={vf1med:.4f}")
        print(f"[SUMMARY-F1 ] {ds_name} | TEST mean={tf1m:.4f} std={tf1s:.4f} median={tf1med:.4f}")

        vaccm, vaccs, vaccmed = res["cv_val_acc"]["mean"], res["cv_val_acc"]["std"], res["cv_val_acc"]["median"]
        taccm, taccs, taccmed = res["cv_test_acc"]["mean"], res["cv_test_acc"]["std"], res["cv_test_acc"]["median"]
        print(f"[SUMMARY-ACC] {ds_name} | VAL mean={vaccm:.4f} std={vaccs:.4f} median={vaccmed:.4f}")
        print(f"[SUMMARY-ACC] {ds_name} | TEST mean={taccm:.4f} std={taccs:.4f} median={taccmed:.4f}")

    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] Metrics -> {out}")

    print("\n[Done] Genomic Benchmarks evaluation completed.")


if __name__ == "__main__":
    main()
