# ATLAS: A Two-Stage Genomic Language Model for Cross-Species Regulatory Modeling

## Introduction
![ATLAS overview](figs/model.pdf)

ATLAS is a two-stage genomic language model for cross-species regulatory modeling, aiming to learn representations that are both species-invariant and functionally grounded for regulatory prediction. In the Cross-species Foundational Representation (CFR) stage, ATLAS is pretrained with masked language modeling (MLM) on a large multi-species genomic corpus to capture conserved regulatory syntax and transferable sequence priors. In the Functional Representation Enhancement (FRE) stage, ATLAS initializes from the CFR backbone and aligns representations with regulatory phenotypes via supervised sequence-to-signal pretraining on multi-track epigenomic readouts, while using an auxiliary MLM objective to preserve sequence modeling ability. 

## Quick Start

### 1) Create a Conda environment

```bash
conda create -n atlas python=3.10 -y
conda activate atlas
pip install -r requirements.txt
```

### 2) Data Processing

We prepare data in two stages:
- **CFR**: multi-species RefSeq genomes.
- **FRE**: sequence-to-signal data.

#### CFR Stage
```bash
# 1) Download RefSeq assemblies
cd data/pretraining_data/multi-species
bash download_refseq_genomes.sh
# outputs: ./refseq_genomes/<species>_<accession>/...

# 2) Slice genomes into fixed-length windows and shard
python build_nt_like_fasta_shards.py \
  --fasta <PATH_TO_GENOME_FASTA(.gz)> \
  --out_dir <OUT_DIR_FOR_SHARDS> \
  --window_bp 6100 \
  --overlap_bp 50 \
  --max_n_frac 0.05 \
  --records_per_shard 20000
```
#### FRE Stage
**Directory layout (recommended)**

- `genomic_profile/`
  - `manifests/epimap_tracks.tsv`
  - `tracks/epimap_avg/`              # downloaded bigWigs
  - `refs/`                           # hg19 fasta + chrom sizes
  - `epimap_npz/`                     # output shards (train/val/test)
  - `download_manifest.sh`
  - `build_epimap_npz_shards.py`

```bash
# 1) Download EpiMap bigWig tracks
cd data/pretraining_data/genomic_profile
bash download_manifest.sh

# 2) Build offline NPZ shards
python build_epimap_npz_shards.py \
  --hg19_fasta refs/hg19.fa \
  --hg19_chrom_sizes refs/hg19.chrom.sizes \
  --tracks_dir tracks/epimap_avg \
  --epimap_tracks_tsv manifests/epimap_tracks.tsv \
  --out_dir epimap_npz \
  --seq_len_bp 6144 \
  --bin_bp 96 \
  --log1p \
  --clip_quantile 0.999 \
  --clip_samples 2000 \
  --shard_size 512 \
  --y_dtype float16 \
  --seed 42
```
### 3) Model Pretraining
#### CFR Stage Pretraining
In the CFR stage, we pretrain the DNA encoder on multi-species genome sequences using MLM.
```bash
# we use NT tokenizer
python train_student.py \
  --positive_dir <OUT_DIR_FOR_SHARDS> \
  --teacher_dir  src/hf/nucleotide-transformer-v2-500m-multi-species \   
  --pretrain_mode mlm \
  --seq_len 2048 \
  --batch_size 8 \
  --epochs 20 \
  --amp bf16 \
  --save_path config/cfr.pt
```
#### FRE Stage Pretraining
In the FRE stage, the model is jointly pretrained with supervised multi-track profile regression and an auxiliary MLM objective. The training script supports dense-to-MoE upcycling, where a dense CFR checkpoint is loaded and the attention MLP can be copied into each MoE expert for an initialization.  
**Inputs**
- `--npz_dir`: directory containing NPZ shards with:
  - `seq`: ASCII-encoded DNA string (fixed bp length)
  - `y`: binned targets with shape `[T_bins, C]`  
- `--teacher_dir`: tokenizer directory (NT-v2 tokenizer)
- `--resume_path`: CFR stage checkpoint to initialize the backbone
- `--n_tracks`: number of target tracks (channels) in `y`
- `--save_path`: output checkpoint path

```bash
python train_profile_moe.py \
  --npz_dir /path/to/genomic_profile/epimap_npz \
  --teacher_dir /path/to/tokenizer \
  --resume_path /path/to/stage1_checkpoint.pt \
  --save_path /path/to/stage2_fre_moe.pt \
  --n_tracks 192 \
  --use_moe \
  --moe_num_experts 8 \
  --moe_top_k 2 \
  --moe_aux_weight 1e-2 \
  --mlm_weight 0.2
```
### 4) Model Evaluation
In the evaluation process, the model is assessed using metrics such as MCC, ACC, and Macro-F1. The evaluation script allows for the comparison of different model configurations (e.g., linear probe, LoRA, full fine-tuning) across K-fold cross-validation.
#### NT Benchmark
```bash
python eval_NT_benchmark.py \
  --datasets_root /path/to/NT_benchmark \
  --encoder_type student \
  --student_ckpt /path/to/checkpoint.pt \
  --student_tokenizer /path/to/tokenizer \
  --mode full_finetune \
  --metrics_out /path/to/results.json \
  --batch_size 16 \
  --k_folds 10 \
  --max_len 512 \
  --epochs 10 \
  --lr 2e-5 \
  --amp
```
#### Genomic Benchmark
```bash
python eval_genomic_benchmarks.py \
  --datasets_root /path/to/genomic_benchmarks \
  --encoder_type student \
  --student_ckpt /path/to/checkpoint.pt \
  --student_tokenizer /path/to/tokenizer \
  --mode full_finetune \
  --metrics_out /path/to/results.json \
  --batch_size 16 \
  --k_folds 5 \
  --max_len 512 \
  --epochs 10 \
  --lr 2e-5 \
  --amp
```
#### GUE Benchmark
```bash
python eval_GUE_benchmark.py \
  --datasets_root /path/to/GUE_benchmark \
  --encoder_type student \
  --student_ckpt /path/to/checkpoint.pt \
  --student_tokenizer /path/to/tokenizer \
  --mode full_finetune \
  --metrics_out /path/to/results.json \
  --batch_size 16 \
  --max_len 512 \
  --epochs 10 \
  --lr 2e-5 \
  --amp
```










