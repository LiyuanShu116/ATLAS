#!/usr/bin/env bash
set -euo pipefail

# Check if NCBI datasets CLI is installed
# Install guide: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/
if ! command -v datasets &>/dev/null; then
  echo "Error: 'datasets' command not found."
  echo "Please install NCBI Datasets CLI first (ncbi-datasets-cli)."
  echo "See: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/"
  exit 1
fi

# Output directory for all downloads
OUT_DIR="refseq_genomes"
mkdir -p "${OUT_DIR}"

# You can change what to include: genome,gff3,cds,protein,genome,seq-report,annotation-report, etc.
INCLUDE_FILES="genome,gff3"

# Declare species -> RefSeq accession map
declare -A ACCESSIONS

# Mammals
ACCESSIONS["Mus_musculus"]="GCF_000001635.27"          # mouse
ACCESSIONS["Rattus_norvegicus"]="GCF_015227675.2"      # rat
ACCESSIONS["Macaca_mulatta"]="GCF_003339765.1"         # rhesus macaque
ACCESSIONS["Canis_lupus_familiaris"]="GCF_000002285.5" # dog
ACCESSIONS["Bos_taurus"]="GCF_002263795.3"             # cattle
ACCESSIONS["Sus_scrofa"]="GCF_000003025.6"             # pig

# Non-mammalian vertebrates
ACCESSIONS["Danio_rerio"]="GCF_000002035.6"            # zebrafish
ACCESSIONS["Gallus_gallus"]="GCF_016699485.2"          # chicken

# Invertebrate model
ACCESSIONS["Drosophila_melanogaster"]="GCF_000001215.4"  # fruit fly

# Yeasts and fungi
ACCESSIONS["Saccharomyces_cerevisiae"]="GCF_000146045.2" # budding yeast
ACCESSIONS["Schizosaccharomyces_pombe"]="GCF_000002945.1" # fission yeast

# Bacteria
ACCESSIONS["Escherichia_coli_K12_MG1655"]="GCF_000005845.2" # E. coli K-12 MG1655
ACCESSIONS["Bacillus_subtilis_168"]="GCF_000009045.1"       # B. subtilis 168

# Loop over all species and download + unzip
for species in "${!ACCESSIONS[@]}"; do
  acc="${ACCESSIONS[$species]}"
  zip_path="${OUT_DIR}/${species}_${acc}.zip"
  dest_dir="${OUT_DIR}/${species}_${acc}"

  echo "============================================================"
  echo "Downloading ${species} (${acc})..."
  echo "  Output zip:  ${zip_path}"
  echo "  Output dir:  ${dest_dir}"

  # Download zip if it does not exist yet
  if [[ -f "${zip_path}" ]]; then
    echo "  Zip file already exists, skip download."
  else
    datasets download genome accession "${acc}" \
      --include "${INCLUDE_FILES}" \
      --filename "${zip_path}" \
      --no-progressbar
  fi

  # Unzip to a dedicated directory
  if [[ -d "${dest_dir}" ]]; then
    echo "  Unzipped directory already exists, skip unzip."
  else
    echo "  Unzipping..."
    mkdir -p "${dest_dir}"
    unzip -q "${zip_path}" -d "${dest_dir}"
    echo "  Done."
  fi
done

echo "============================================================"
echo "All downloads finished. Output root directory: ${OUT_DIR}"
