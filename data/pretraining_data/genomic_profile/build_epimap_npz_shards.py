#!/usr/bin/env python3
# build_epimap_npz_shards.py
# Convert EpiMap bigWig tracks into binned targets and save offline NPZ shards.
# English comments only.

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm


N_CODE = ord("N")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_chrom_sizes(chrom_sizes_path: str) -> Dict[str, int]:
    """
    Load hg19 chrom sizes and keep only chr1-22, chrX, chrY.
    """
    sizes = {}
    keep = re.compile(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$")  # chr1-22, chrX, chrY
    with open(chrom_sizes_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            chrom, n = line.rstrip("\n").split("\t")[:2]
            if not keep.match(chrom):
                continue
            sizes[chrom] = int(n)
    if not sizes:
        raise ValueError(f"No valid chromosomes loaded from {chrom_sizes_path}")
    return sizes


def open_bigwigs(track_manifest: pd.DataFrame, tracks_dir: str) -> List[pyBigWig.pyBigWig]:
    """
    track_manifest must have at least a 'filename' column.
    """
    bws: List[pyBigWig.pyBigWig] = []
    for fn in track_manifest["filename"].tolist():
        path = os.path.join(tracks_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing bigWig: {path}")
        bws.append(pyBigWig.open(path))
    return bws


def compute_common_chrom_sizes(
    ref_chrom_sizes: Dict[str, int],
    bws: List[pyBigWig.pyBigWig],
) -> Dict[str, int]:
    """
    Compute chromosome sizes that are safe across ALL bigWigs.
    For each chromosome, use the minimum length among:
      - reference chrom size
      - each bigWig chrom size
    Only keep chromosomes present in ALL bigWigs.
    """
    if not bws:
        raise ValueError("No bigWigs opened.")

    bw_chrom_dicts = [bw.chroms() for bw in bws]  # dict: chrom -> length
    common = set(ref_chrom_sizes.keys())
    for d in bw_chrom_dicts:
        common &= set(d.keys())

    if not common:
        example = list(bw_chrom_dicts[0].keys())[:10]
        raise ValueError(
            "No common chromosomes between reference and bigWigs. "
            "Likely chromosome naming mismatch (e.g., chr1 vs 1).\n"
            f"Example bigWig chroms: {example}"
        )

    safe_sizes = {}
    for c in sorted(common):
        m = ref_chrom_sizes[c]
        for d in bw_chrom_dicts:
            m = min(m, int(d[c]))
        safe_sizes[c] = m

    return safe_sizes


def bw_binned_stats(
    bw: pyBigWig.pyBigWig,
    chrom: str,
    start: int,
    end: int,
    n_bins: int,
    stat: str = "mean",
) -> np.ndarray:
    """
    Safe bigWig binning. If chrom is missing or bounds are invalid, return zeros.
    """
    chroms = bw.chroms()
    if chrom not in chroms:
        return np.zeros((n_bins,), dtype=np.float32)

    L = int(chroms[chrom])
    if start < 0:
        start = 0
    if end > L:
        end = L
    if end <= start:
        return np.zeros((n_bins,), dtype=np.float32)

    try:
        vals = bw.stats(chrom, start, end, nBins=n_bins, type=stat)
    except RuntimeError:
        return np.zeros((n_bins,), dtype=np.float32)

    x = np.array([0.0 if v is None else float(v) for v in vals], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x[x < 0] = 0.0
    return x


def sample_one_window(
    chrom_sizes: Dict[str, int],
    chroms: List[str],
    weights: np.ndarray,
    seq_len_bp: int,
) -> Tuple[str, int, int]:
    """
    Sample a single random window, proportional to chromosome sizes.
    """
    chrom = str(np.random.choice(chroms, p=weights))
    L = int(chrom_sizes[chrom])
    start = random.randint(0, L - seq_len_bp - 1)
    end = start + seq_len_bp
    return chrom, start, end


def iter_bed_windows(bed_path: str, seq_len_bp: int) -> Iterator[Tuple[str, int, int]]:
    """
    Yield BED3 windows. If an interval length != seq_len_bp, it is skipped.
    """
    with open(bed_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, s, e = line.rstrip("\n").split("\t")[:3]
            s = int(s)
            e = int(e)
            if e - s != seq_len_bp:
                continue
            yield chrom, s, e


def seq_to_u8(seq: str) -> np.ndarray:
    """
    Convert uppercase DNA string to uint8 ASCII array.
    """
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8)


def n_fraction(seq_u8: np.ndarray) -> float:
    """
    Fraction of 'N' bases in a window.
    """
    if seq_u8.size == 0:
        return 1.0
    return float((seq_u8 == N_CODE).mean())


def estimate_per_track_clip(
    bws: List[pyBigWig.pyBigWig],
    fa: Fasta,
    chrom_sizes: Dict[str, int],
    chroms: List[str],
    seq_len_bp: int,
    n_bins: int,
    stat: str,
    clip_quantile: float,
    max_samples: int,
    max_n_frac: float,
) -> np.ndarray:
    """
    Estimate robust clip thresholds per track using random windows that pass N-filter.
    IMPORTANT: clip is computed on bin-level values (not per-window maxima).
    """
    if max_samples <= 0:
        raise ValueError("clip_samples must be > 0")

    weights = np.array([chrom_sizes[c] for c in chroms], dtype=np.float64)
    weights /= weights.sum()

    # Collect bin-level values across windows.
    # Shape: [max_samples * n_bins, n_tracks]
    # Note: for n_bins=128, max_samples=2000, n_tracks=192, this is ~196MB float32.
    val_mat = np.zeros((max_samples * n_bins, len(bws)), dtype=np.float32)

    accepted = 0
    tries = 0
    max_tries = max_samples * 50  # hard cap

    pbar = tqdm(total=max_samples, desc="clip-est", dynamic_ncols=True)
    while accepted < max_samples and tries < max_tries:
        tries += 1
        chrom, s, e = sample_one_window(chrom_sizes, chroms, weights, seq_len_bp)
        try:
            seq = fa[chrom][s:e]
        except KeyError:
            continue
        if seq is None or len(seq) != seq_len_bp:
            continue
        su8 = seq_to_u8(seq)
        if n_fraction(su8) > max_n_frac:
            continue

        row0 = accepted * n_bins
        row1 = row0 + n_bins
        for t, bw in enumerate(bws):
            x = bw_binned_stats(bw, chrom, s, e, n_bins=n_bins, stat=stat)
            val_mat[row0:row1, t] = x

        accepted += 1
        pbar.update(1)

    pbar.close()

    if accepted < max_samples:
        raise RuntimeError(
            f"clip-est: only collected {accepted}/{max_samples} valid windows "
            f"(max_n_frac={max_n_frac}). Consider increasing max_tries_factor, "
            f"relaxing max_n_frac, or reducing clip_samples."
        )

    clip = np.quantile(val_mat, clip_quantile, axis=0).astype(np.float32)
    clip = np.maximum(clip, 1e-6).astype(np.float32)
    return clip


def write_meta_json(
    out_dir: str,
    args: argparse.Namespace,
    n_bins: int,
    n_tracks: int,
    track_ids: List[str],
    train_chroms: List[str],
    val_chroms: List[str],
    test_chroms: List[str],
    clip: np.ndarray,
    clip_estimation_mode: str,
    clip_estimation_windows: int,
) -> None:
    """Write a meta.json file recording preprocessing parameters."""
    meta = {
        "script": "build_epimap_npz_shards.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "derived": {
            "n_bins": int(n_bins),
            "n_tracks": int(n_tracks),
            "track_ids": list(track_ids),
            "train_chroms": list(train_chroms),
            "val_chroms": list(val_chroms),
            "test_chroms": list(test_chroms),
            "clip_estimation_mode": str(clip_estimation_mode),
            "clip_estimation_windows": int(clip_estimation_windows),
            "clip": [float(x) if np.isfinite(x) else float("inf") for x in clip.astype(np.float32)],
        },
    }
    out_path = os.path.join(out_dir, "meta.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def save_shard_npz(
    out_path: str,
    seq_u8: np.ndarray,
    y: np.ndarray,
    coord: List[str],
    track_ids: List[str],
) -> None:
    """
    seq_u8: [B, seq_len_bp] uint8 ASCII codes
    y:      [B, n_bins, n_tracks] float32/float16
    coord:  list of strings length B
    """
    np.savez_compressed(
        out_path,
        seq=seq_u8,
        y=y,
        coord=np.array(coord, dtype=object),
        track_ids=np.array(track_ids, dtype=object),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hg19_fasta", required=True, help="hg19 fasta file (indexed by pyfaidx).")
    ap.add_argument("--hg19_chrom_sizes", required=True, help="hg19.chrom.sizes")
    ap.add_argument("--tracks_dir", required=True, help="Directory containing downloaded EpiMap bigWigs.")
    ap.add_argument("--epimap_tracks_tsv", required=True, help="Your epimap_tracks.tsv")
    ap.add_argument("--out_dir", required=True)

    # Window and binning
    ap.add_argument("--seq_len_bp", type=int, required=True, help="Sequence window length in bp.")
    ap.add_argument("--bin_bp", type=int, default=96, help="Bin size in bp (recommended 96 for 6-mer alignment).")
    ap.add_argument("--stat", choices=["mean", "sum"], default="mean", help="Binning statistic for bigWig.stats.")

    # Tokenizer alignment safeguards
    ap.add_argument("--require_div_by6", action="store_true", help="Require seq_len_bp divisible by 6 (non-overlap 6-mer).")

    # Split strategy (chrom-based random sampling)
    ap.add_argument("--train_chroms", type=str, default=None, help="Comma-separated chromosomes for train. Default: all except val/test.")
    ap.add_argument("--val_chroms", type=str, default="chr10", help="Comma-separated chromosomes for val.")
    ap.add_argument("--test_chroms", type=str, default="chr8,chr9", help="Comma-separated chromosomes for test.")
    ap.add_argument("--n_train", type=int, default=200000)
    ap.add_argument("--n_val", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=10000)

    # Alternative: use BED windows instead of random sampling
    ap.add_argument("--train_bed", type=str, default=None)
    ap.add_argument("--val_bed", type=str, default=None)
    ap.add_argument("--test_bed", type=str, default=None)

    # Sequence filtering
    ap.add_argument("--max_n_frac", type=float, default=0.05, help="Skip windows with N fraction > this threshold.")
    ap.add_argument("--max_tries_factor", type=int, default=50, help="Max sampling tries = target_n * factor (random mode).")

    # Target transforms
    ap.add_argument("--log1p", action="store_true", help="Apply log1p to binned targets (recommended).")
    ap.add_argument("--clip_quantile", type=float, default=0.999, help="Per-track clip quantile.")
    ap.add_argument("--clip_samples", type=int, default=2000, help="Windows used to estimate clip. Reduce if many tracks.")
    ap.add_argument("--no_clip", action="store_true", help="Disable clipping.")

    # Sharding
    ap.add_argument("--shard_size", type=int, default=512)
    ap.add_argument("--y_dtype", choices=["float32", "float16"], default="float16")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    if args.require_div_by6 and (args.seq_len_bp % 6 != 0):
        raise ValueError(f"seq_len_bp ({args.seq_len_bp}) must be divisible by 6 for non-overlapping 6-mer tokenization.")

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

    if args.seq_len_bp % args.bin_bp != 0:
        raise ValueError(f"seq_len_bp ({args.seq_len_bp}) must be divisible by bin_bp ({args.bin_bp}).")
    n_bins = args.seq_len_bp // args.bin_bp
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    # Load track manifest
    tracks = pd.read_csv(args.epimap_tracks_tsv, sep="\t")
    if "filename" not in tracks.columns:
        raise ValueError("epimap_tracks.tsv must contain a 'filename' column matching local bigWig filenames.")
    if "track_id" in tracks.columns:
        track_ids = tracks["track_id"].astype(str).tolist()
    else:
        track_ids = tracks["filename"].astype(str).tolist()

    # Open genome
    fa = Fasta(args.hg19_fasta, as_raw=True, sequence_always_upper=True)

    # Open bigWigs
    bws = open_bigwigs(tracks, args.tracks_dir)

    # Restrict to chromosomes present in ALL bigWigs, clamp lengths to min
    ref_chrom_sizes = load_chrom_sizes(args.hg19_chrom_sizes)
    chrom_sizes = compute_common_chrom_sizes(ref_chrom_sizes, bws)
    print(f"[OK] safe chromosomes: {len(chrom_sizes)} (ref={len(ref_chrom_sizes)})")

    all_chroms = sorted(chrom_sizes.keys())
    safe_set = set(all_chroms)

    # Parse split chrom lists
    val_chroms = [c.strip() for c in args.val_chroms.split(",") if c.strip()]
    test_chroms = [c.strip() for c in args.test_chroms.split(",") if c.strip()]
    val_chroms = [c for c in val_chroms if c in safe_set]
    test_chroms = [c for c in test_chroms if c in safe_set]

    if args.train_chroms is None:
        holdout = set(val_chroms + test_chroms)
        train_chroms = [c for c in all_chroms if c not in holdout]
    else:
        train_chroms = [c.strip() for c in args.train_chroms.split(",") if c.strip()]
        train_chroms = [c for c in train_chroms if c in safe_set]

    if len(train_chroms) == 0:
        raise ValueError("No training chromosomes after filtering to safe chromosomes.")
    if len(val_chroms) == 0:
        print("[WARN] No validation chromosomes left after filtering; val sampling may fail unless you set --val_chroms.")
    if len(test_chroms) == 0:
        print("[WARN] No test chromosomes left after filtering; test sampling may fail unless you set --test_chroms.")

    # Estimate clip thresholds from TRAIN distribution.
    # We use bin-level values (not per-window maxima) for robust clipping.
    if args.no_clip:
        clip = np.full((len(bws),), np.inf, dtype=np.float32)
        clip_estimation_mode = "disabled"
        clip_estimation_windows = 0
    else:
        # If train_bed is provided, prefer using it for clip-est (but still filter by N).
        if args.train_bed:
            bed_iter = iter_bed_windows(args.train_bed, args.seq_len_bp)
            # Collect first args.clip_samples valid windows (pass N-filter)
            use: List[Tuple[str, int, int]] = []
            for chrom, s, e in bed_iter:
                if chrom not in chrom_sizes:
                    continue
                try:
                    seq = fa[chrom][s:e]
                except KeyError:
                    continue
                if seq is None or len(seq) != args.seq_len_bp:
                    continue
                su8 = seq_to_u8(seq)
                if n_fraction(su8) > args.max_n_frac:
                    continue
                use.append((chrom, s, e))
                if len(use) >= args.clip_samples:
                    break
            if len(use) < max(10, min(100, args.clip_samples)):
                raise RuntimeError(
                    f"clip-est: too few valid windows from train_bed: {len(use)}. "
                    f"Relax max_n_frac or provide more windows."
                )
            # Compute clip using bin-level values from the collected windows.
            val_mat = np.zeros((len(use) * n_bins, len(bws)), dtype=np.float32)
            for i, (chrom, s, e) in enumerate(tqdm(use, desc="clip-est", dynamic_ncols=True)):
                row0 = i * n_bins
                row1 = row0 + n_bins
                for t, bw in enumerate(bws):
                    x = bw_binned_stats(bw, chrom, s, e, n_bins=n_bins, stat=args.stat)
                    val_mat[row0:row1, t] = x
            clip = np.quantile(val_mat, args.clip_quantile, axis=0).astype(np.float32)
            clip = np.maximum(clip, 1e-6).astype(np.float32)
            clip_estimation_mode = "bed"
            clip_estimation_windows = int(len(use))
        else:
            clip = estimate_per_track_clip(
                bws=bws,
                fa=fa,
                chrom_sizes=chrom_sizes,
                chroms=train_chroms,
                seq_len_bp=args.seq_len_bp,
                n_bins=n_bins,
                stat=args.stat,
                clip_quantile=args.clip_quantile,
                max_samples=args.clip_samples,
                max_n_frac=args.max_n_frac,
            )
            clip_estimation_mode = "random"
            clip_estimation_windows = int(args.clip_samples)
    print(f"[OK] clip thresholds computed: shape={clip.shape}")

    y_dtype = np.float16 if args.y_dtype == "float16" else np.float32

    def build_split_random(split: str, chroms: List[str], target_n: int) -> None:
        """
        Randomly sample windows and ensure we write exactly target_n examples
        (subject to max_tries_factor).
        """
        out_split = os.path.join(args.out_dir, split)
        weights = np.array([chrom_sizes[c] for c in chroms], dtype=np.float64)
        weights /= weights.sum()

        shard_idx = 0
        buf_seq: List[np.ndarray] = []
        buf_y: List[np.ndarray] = []
        buf_coord: List[str] = []

        accepted = 0
        tries = 0
        max_tries = max(1, target_n * args.max_tries_factor)

        pbar = tqdm(total=target_n, desc=f"build-{split}", dynamic_ncols=True)
        while accepted < target_n and tries < max_tries:
            tries += 1
            chrom, s, e = sample_one_window(chrom_sizes, chroms, weights, args.seq_len_bp)

            try:
                seq = fa[chrom][s:e]
            except KeyError:
                continue
            if seq is None or len(seq) != args.seq_len_bp:
                continue

            seq_u8 = seq_to_u8(seq)
            if seq_u8.shape[0] != args.seq_len_bp:
                continue

            # Filter out gap-like windows
            if n_fraction(seq_u8) > args.max_n_frac:
                continue

            # Targets: [n_bins, n_tracks]
            y = np.zeros((n_bins, len(bws)), dtype=np.float32)
            for t, bw in enumerate(bws):
                x = bw_binned_stats(bw, chrom, s, e, n_bins=n_bins, stat=args.stat)
                if np.isfinite(clip[t]):
                    x = np.minimum(x, clip[t])
                if args.log1p:
                    x = np.log1p(x)
                y[:, t] = x

            buf_seq.append(seq_u8)
            buf_y.append(y.astype(y_dtype, copy=False))
            buf_coord.append(f"{chrom}:{s}-{e}")
            accepted += 1
            pbar.update(1)

            if len(buf_seq) >= args.shard_size:
                out_path = os.path.join(out_split, f"{split}-{shard_idx:05d}.npz")
                save_shard_npz(
                    out_path=out_path,
                    seq_u8=np.stack(buf_seq, axis=0),
                    y=np.stack(buf_y, axis=0),
                    coord=buf_coord,
                    track_ids=track_ids,
                )
                shard_idx += 1
                buf_seq, buf_y, buf_coord = [], [], []

        pbar.close()

        if accepted < target_n:
            raise RuntimeError(
                f"[FAIL] {split}: only generated {accepted}/{target_n} examples "
                f"after {tries} tries (max_n_frac={args.max_n_frac}). "
                f"Try increasing --max_tries_factor, relaxing --max_n_frac, "
                f"or reducing target_n."
            )

        # Flush tail
        if buf_seq:
            out_path = os.path.join(out_split, f"{split}-{shard_idx:05d}.npz")
            save_shard_npz(
                out_path=out_path,
                seq_u8=np.stack(buf_seq, axis=0),
                y=np.stack(buf_y, axis=0),
                coord=buf_coord,
                track_ids=track_ids,
            )

        print(f"[OK] {split}: wrote shards to {out_split} (n={accepted})")

    def build_split_bed(split: str, bed_path: str) -> None:
        """
        Build shards from a BED file. We will skip invalid windows / high-N windows,
        so the final number may be smaller than the BED count.
        """
        out_split = os.path.join(args.out_dir, split)
        shard_idx = 0

        buf_seq: List[np.ndarray] = []
        buf_y: List[np.ndarray] = []
        buf_coord: List[str] = []

        accepted = 0
        skipped_n = 0
        skipped_other = 0

        pbar = tqdm(iter_bed_windows(bed_path, args.seq_len_bp), desc=f"build-{split}", dynamic_ncols=True)
        for chrom, s, e in pbar:
            if chrom not in chrom_sizes:
                skipped_other += 1
                continue
            try:
                seq = fa[chrom][s:e]
            except KeyError:
                skipped_other += 1
                continue
            if seq is None or len(seq) != args.seq_len_bp:
                skipped_other += 1
                continue

            seq_u8 = seq_to_u8(seq)
            if n_fraction(seq_u8) > args.max_n_frac:
                skipped_n += 1
                continue

            y = np.zeros((n_bins, len(bws)), dtype=np.float32)
            for t, bw in enumerate(bws):
                x = bw_binned_stats(bw, chrom, s, e, n_bins=n_bins, stat=args.stat)
                if np.isfinite(clip[t]):
                    x = np.minimum(x, clip[t])
                if args.log1p:
                    x = np.log1p(x)
                y[:, t] = x

            buf_seq.append(seq_u8)
            buf_y.append(y.astype(y_dtype, copy=False))
            buf_coord.append(f"{chrom}:{s}-{e}")
            accepted += 1

            if len(buf_seq) >= args.shard_size:
                out_path = os.path.join(out_split, f"{split}-{shard_idx:05d}.npz")
                save_shard_npz(
                    out_path=out_path,
                    seq_u8=np.stack(buf_seq, axis=0),
                    y=np.stack(buf_y, axis=0),
                    coord=buf_coord,
                    track_ids=track_ids,
                )
                shard_idx += 1
                buf_seq, buf_y, buf_coord = [], [], []

        if buf_seq:
            out_path = os.path.join(out_split, f"{split}-{shard_idx:05d}.npz")
            save_shard_npz(
                out_path=out_path,
                seq_u8=np.stack(buf_seq, axis=0),
                y=np.stack(buf_y, axis=0),
                coord=buf_coord,
                track_ids=track_ids,
            )

        print(
            f"[OK] {split}: wrote shards to {out_split} (accepted={accepted}, "
            f"skipped_n={skipped_n}, skipped_other={skipped_other})"
        )

    # Build splits
    if args.train_bed:
        build_split_bed("train", args.train_bed)
    else:
        build_split_random("train", train_chroms, args.n_train)

    if args.val_bed:
        build_split_bed("val", args.val_bed)
    else:
        build_split_random("val", val_chroms, args.n_val)

    if args.test_bed:
        build_split_bed("test", args.test_bed)
    else:
        build_split_random("test", test_chroms, args.n_test)

    # Record preprocessing parameters and clip thresholds.
    write_meta_json(
        out_dir=args.out_dir,
        args=args,
        n_bins=n_bins,
        n_tracks=len(bws),
        track_ids=track_ids,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        clip=clip,
        clip_estimation_mode=clip_estimation_mode,
        clip_estimation_windows=clip_estimation_windows,
    )

    for bw in bws:
        bw.close()

    print("[DONE]")


if __name__ == "__main__":
    main()
