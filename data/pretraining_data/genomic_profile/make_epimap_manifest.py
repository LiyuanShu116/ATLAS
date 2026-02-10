#!/usr/bin/env python3
# Generate a track manifest from the EpiMap "averagetracks_pergroup" directory listing.
# Comments are in English.

import argparse
import re
import sys
from urllib.request import urlopen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="https://epigenome.wustl.edu/epimap/data/averagetracks_pergroup/")
    ap.add_argument("--assays", required=True, help="Comma-separated, e.g., ATAC-seq,DNase-seq,H3K27ac,H3K4me1,H3K4me3,CTCF")
    ap.add_argument("--groups", required=True, help="Comma-separated tissue groups, e.g., Brain,Liver,BloodandT-cell")
    ap.add_argument("--mode", choices=["observed", "imputed"], default="imputed")
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    assays = set([x.strip() for x in args.assays.split(",") if x.strip()])
    groups = set([x.strip() for x in args.groups.split(",") if x.strip()])

    import os
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)

    html = urlopen(args.base_url).read().decode("utf-8", errors="ignore")

    # Example filename: average_ATAC-seq_imputed_Brain.bigWig
    pat = re.compile(r'href="(average_([^_]+)_(observed|imputed)_([^"]+)\.bigWig)"')
    rows = []
    for m in pat.finditer(html):
        fname = m.group(1)
        assay = m.group(2)
        mode = m.group(3)
        group = m.group(4)
        if assay in assays and group in groups and mode == args.mode:
            url = args.base_url.rstrip("/") + "/" + fname
            track_id = f"{assay}__{mode}__{group}"
            rows.append((track_id, assay, group, mode, url, fname))

    if not rows:
        print("No tracks matched. Check assay/group strings against the directory listing.", file=sys.stderr)
        sys.exit(1)

    with open(args.out_tsv, "w") as f:
        f.write("track_id\tassay\tgroup\tmode\turl\tfilename\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    print(f"[OK] wrote {args.out_tsv} with {len(rows)} tracks")

if __name__ == "__main__":
    main()
