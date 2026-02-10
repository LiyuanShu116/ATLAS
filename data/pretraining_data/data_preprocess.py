#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import gzip
import io
import math
import random

# ------------------ FASTA streaming helpers ------------------ #
def open_maybe_gzip(path: Path):
    # Open text file seamlessly whether it's plain or .gz
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")

def iter_fasta(path: Path):
    # Stream FASTA records as (header, sequence) tuples
    with open_maybe_gzip(path) as f:
        header, buf = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(buf)
                header, buf = line[1:], []
            else:
                buf.append(line)
        if header is not None:
            yield header, "".join(buf)

# ------------------ Cleaning and slicing ------------------ #
IUPAC_TO_ACGTN = str.maketrans({
    # Canonical bases
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    # Ambiguity codes -> N
    "R": "N", "Y": "N", "S": "N", "W": "N", "K": "N", "M": "N",
    "B": "N", "D": "N", "H": "N", "V": "N", "N": "N"
})
def clean_to_acgtn(seq: str) -> str:
    # Uppercase and map IUPAC ambiguity to 'N'
    return seq.upper().translate(IUPAC_TO_ACGTN)

def is_primary_chrom(header: str, allow_mt: bool = False) -> bool:
    """
    Heuristic filter to keep GRCh38 primary autosomes + sex chromosomes.
    Accepts headers like:
      'NC_000001.11 Homo sapiens chromosome 1, GRCh38.p13 Primary Assembly'
    """
    h = header
    if "Primary Assembly" not in h and "GRCh38" not in h:
        return False
    m = re.search(r"chromosome\s+([0-9XYM]+)", h, flags=re.IGNORECASE)
    if not m:
        return False
    chrom = m.group(1).upper()
    if chrom == "M":
        return allow_mt
    if chrom in {"X", "Y"}:
        return True
    try:
        k = int(chrom)
        return 1 <= k <= 22
    except ValueError:
        return False

def windows(start: int, end: int, win: int, step: int):
    # Yield half-open [s, e) windows with step size
    s = start
    while s + win <= end:
        yield s, s + win
        s += step

# ------------------ Sharding writer ------------------ #
class FastaShardWriter:
    def __init__(self, out_dir: Path, prefix: str = "shard", records_per_shard: int = 20000):
        self.out_dir = out_dir
        self.prefix = prefix
        self.records_per_shard = records_per_shard
        self.count = 0
        self.shard_idx = 0
        self.handle = None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _roll(self):
        if self.handle:
            self.handle.close()
        fname = f"{self.prefix}_{self.shard_idx:05d}.fa"
        self.handle = (self.out_dir / fname).open("w", encoding="utf-8")
        self.shard_idx += 1
        self.count = 0

    def write(self, header: str, seq: str):
        if self.handle is None or self.count >= self.records_per_shard:
            self._roll()
        self.handle.write(f">{header}\n")
        # Wrap sequence to 80 columns for readability (optional)
        for i in range(0, len(seq), 80):
            self.handle.write(seq[i:i+80] + "\n")
        self.count += 1

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

# ------------------ Main pipeline ------------------ #
def build(args):
    fasta_path = Path(args.fasta)
    out_root = Path(args.out_dir)
    out_pos = out_root
    # out_pos.mkdir(parents=True, exist_ok=True)

    writer = FastaShardWriter(out_pos, prefix="drosophila", records_per_shard=args.records_per_shard)

    kept, skipped_short, skipped_n, total = 0, 0, 0, 0
    win = args.window_bp
    step = args.window_bp - args.overlap_bp
    assert step > 0, "overlap_bp must be smaller than window_bp"

    for hdr, raw_seq in iter_fasta(fasta_path):
        total += 1
        if args.only_primary and not is_primary_chrom(hdr, allow_mt=args.include_mt):
            continue

        seq = clean_to_acgtn(raw_seq)
        L = len(seq)
        if L < win:
            continue

        # Slice into 6,100 bp windows with 50 bp overlap (NT-like)
        for s, e in windows(0, L, win, step):
            chunk = seq[s:e]
            if len(chunk) < win:
                skipped_short += 1
                continue

            # Filter by N fraction
            n_frac = chunk.count("N") / float(len(chunk))
            if n_frac > args.max_n_frac:
                skipped_n += 1
                continue

            # Write FASTA record; header carries contig + 1-based coordinates + strand '+'
            rec_header = f"id-{hdr.replace(' ', '_')}:{s+1}-{e}\t+\tLen={len(chunk)}\tLocation={hdr.split()[0]}:{s+1}-{e}"
            writer.write(rec_header, chunk)
            kept += 1

    writer.close()

    print(f"[DONE] kept={kept} skipped_short={skipped_short} skipped_highN={skipped_n}")

def parse_args():
    ap = argparse.ArgumentParser(description="Build NT-like GRCh38 pretraining dataset from a FASTA.")
    ap.add_argument("--fasta", type=str, required=True, help="Path to GRCh38_latest_genomic.fna (plain or .gz).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output root directory; FASTA shards go under out_dir/positive/")
    ap.add_argument("--window_bp", type=int, default=6100, help="Window size in base pairs (NT uses 6,100 bp).")
    ap.add_argument("--overlap_bp", type=int, default=50, help="Overlap between consecutive windows (NT uses 50 bp).")
    ap.add_argument("--max_n_frac", type=float, default=0.05, help="Drop windows with > this fraction of 'N'.")
    ap.add_argument("--only_primary", action="store_true", help="Keep only primary assembly autosomes + X/Y.")
    ap.add_argument("--include_mt", action="store_true", help="Include mitochondrial chromosome if --only_primary is set.")
    ap.add_argument("--records_per_shard", type=int, default=20000, help="How many FASTA records per output shard.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build(args)
