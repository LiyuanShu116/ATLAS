import os
from typing import List, Iterator
from torch.utils.data import Dataset

def _iter_fasta_like(path: str) -> Iterator[str]:
    """
    Stream a FASTA-like file where '>' starts a new record and the following
    lines contain uppercase DNA letters. Header content is ignored; only sequence is returned.
    """
    seq_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_lines:
                    yield "".join(seq_lines).upper().replace(" ", "").replace("\t", "")
                    seq_lines = []
            else:
                seq_lines.append(line)
        if seq_lines:
            yield "".join(seq_lines).upper().replace(" ", "").replace("\t", "")

class PositiveSeqDataset(Dataset):
    """
    Recursively scan a 'positive' directory and collect all sequences from files
    except any filename named 'annotations.jsonl'.
    """
    def __init__(self, root_dir: str):
        super().__init__()
        self.root_dir = root_dir
        self._seqs: List[str] = []
        self._load()

    def _load(self):
        for fname in sorted(os.listdir(self.root_dir)):
            if fname.startswith(".") or fname == "annotations.jsonl":
                continue
            fpath = os.path.join(self.root_dir, fname)
            if not os.path.isfile(fpath):
                continue
            for seq in _iter_fasta_like(fpath):
                # Keep only A/C/G/T/N characters; drop others conservatively
                filtered = "".join(ch for ch in seq if ch in "ACGTN")
                if len(filtered) > 0:
                    self._seqs.append(filtered)

    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, idx):
        return self._seqs[idx]
