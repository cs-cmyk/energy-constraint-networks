# Energy-Based Constraint Networks

**Learning Structural Coherence Across Modalities**

Chirag Shinde · Independent Researcher · [Paper (Zenodo)](https://doi.org/19760813)

---

A modality-agnostic architecture that learns structural coherence from contrastive pairs. The same constraint network detects discourse violations in text and manipulation artifacts in images — the first architecture to transfer across modalities via corruption respecification alone.

## Key Results

**Text domain** (frozen BERT, 7.4M trainable parameters):
- 93.4% accuracy on 6 trained corruption types
- 87.2% accuracy on 9 unseen corruption types the model was never trained on
- Per-position energy decomposition localizes violations within paragraphs

**Vision domain** (frozen DINOv2, 3×3.6M trainable parameters):
- 0.959 AUC on FaceForensics++ Deepfakes
- 0.870±0.019 AUC on Celeb-DF (cross-dataset, zero Celeb-DF training data)
- Three composable branches: structural, frequency, local texture

## How It Works

1. **Frozen encoder** (BERT for text, DINOv2 for vision) produces patch/window-level embeddings
2. **Constraint network** (SSM backbone + dual-head attention) processes embeddings and outputs scalar energy + per-position energy scores
3. **Contrastive training** on coherent inputs (low energy) vs corrupted inputs (high energy)
4. **Corruption = specification**: each corruption type implicitly defines a structural property to enforce. New constraints are added by designing new corruptions — no architecture changes needed

## Architecture

```
Input embeddings → SSM blocks (×6) → Dual-head attention (×2, interleaved) → Energy head
                                      ├─ Head 1: causal masked
                                      └─ Head 2: bidirectional
                                      Single projection per head (no Q/K/V, no KV cache)

Energy: E(x) = mean(e_i) + α · max(e_i)
```

- Half the parameters of standard attention
- No key-value cache — each evaluation is a stateless forward pass
- Per-position energy decomposition enables violation localization

## Repository Structure

```
energy-constraint-networks/
├── text/                          # Text domain
│   ├── model.py                   # Constraint network architecture
│   ├── bert_encoder.py            # BERT token-level encoder (8-token windows)
│   ├── nl_adaptation.py           # Original 6 corruption strategies
│   ├── corruptions_extended.py    # Extended 9 corruption types
│   ├── data.py                    # Synthetic data generator
│   ├── train_bert.py              # Main training script
│   ├── eval_all_corruptions.py    # 15-type generalization evaluation
│   ├── eval_bert.py               # In-distribution evaluation (6 types)
│   ├── eval_showcase.py           # Hand-crafted sentence examples
│   ├── analyze_clustering.py      # Displacement vector analysis
│   └── run_text_ablations.py      # Seed variance + alpha ablation
│
├── vision/                        # Vision domain
│   ├── model.py                   # Constraint network (same architecture)
│   ├── vision_encoder.py          # DINOv2 patch-level encoder
│   ├── vision_corruptions.py      # Face corruption strategies
│   ├── frequency_heatmap.py       # Frequency heatmap rendering
│   ├── dual_pass_encoder.py       # Dual-pass DINOv2 encoder
│   ├── train_paired.py            # Structural branch training
│   ├── train_freq_only.py         # Frequency branch training
│   ├── train_local_only.py        # Local texture branch training
│   ├── eval_combined_final.py     # Three-branch combined evaluation
│   ├── run_experiments.py         # Seed variance + alpha ablation
│   ├── vision_diagnostic.py       # DINOv2 feature diagnostic
│   ├── corruption_alignment.py    # Corruption-artifact alignment test
│   └── create_corruption_pairs.py # Generate corruption training pairs
│
├── data_prep/                     # Data preparation
│   ├── extract_faces.py           # Face extraction from FF++ videos
│   ├── prepare_ffpp.py            # FaceForensics++ download/setup
│   └── split_data.py              # Train/test splitting
│
└── diagrams/                      # Paper figure generation
    └── generare_diagrams.py
```

## Quick Start

### Text Domain

```bash
cd text/

# Install dependencies
pip install torch transformers datasets sentence-transformers mamba-ssm

# Train constraint network (embeddings are pre-computed and cached)
python train_bert.py

# Evaluate on all 15 corruption types (6 trained + 9 unseen)
python eval_all_corruptions.py

# Run displacement vector analysis
python analyze_clustering.py

# Run seed variance and alpha ablation
python run_text_ablations.py --experiment both
```

### Vision Domain

```bash
cd vision/

# Install dependencies
pip install torch torchvision transformers timm mamba-ssm

# Prepare data (requires FaceForensics++ access credentials)
cd ../data_prep/
python prepare_ffpp.py
python extract_faces.py
python split_data.py
cd ../vision/

# Create corruption training pairs
python create_corruption_pairs.py --real_dir /path/to/real/faces

# Train three branches independently
python train_paired.py --real_dir ... --fake_dirs ...      # Structural
python train_freq_only.py --real_dir ... --fake_dirs ...    # Frequency
python train_local_only.py --real_dir ... --fake_dirs ...   # Local texture

# Evaluate combined system
python eval_combined_final.py \
    --struct_checkpoint structural.pt \
    --freq_checkpoint frequency.pt \
    --local_checkpoint local.pt \
    --real_dir /path/to/test/real \
    --fake_dirs /path/to/test/fake

# Run ablation experiments
python run_experiments.py --experiment both ...
```

## Corruption Types

### Text (6 trained + 9 unseen)

| Corruption | Property | Trained | Accuracy |
|-----------|----------|---------|----------|
| Tense shift | Temporal consistency | ✓ | 100% |
| Coref break | Identity consistency | ✓ | 100% |
| Negate | Contradiction | ✓ | 99% |
| Topic splice | Topical coherence | ✓ | 98% |
| Entity swap | Referential integrity | ✓ | 90% |
| Shuffle | Discourse ordering | ✓ | 87% |
| Role/title swap | Role-entity binding | ✗ | 100% |
| Demonstrative | Demonstrative reference | ✗ | 96% |
| Description swap | Definite description | ✗ | 93% |
| Bridging break | Bridging inference | ✗ | 93% |
| Singular/plural | Number agreement | ✗ | 93% |
| Name substitution | Person identity | ✗ | 90% |
| Temporal break | Temporal consistency | ✗ | 88% |
| Number change | Numerical consistency | ✗ | 77% |
| Repetition | Discourse progression | ✗ | 44% |

### Vision

| Corruption | Property |
|-----------|----------|
| Texture splice | Regional compatibility |
| Lighting shift | Illumination consistency |
| Localized smoothing | Texture quality uniformity |
| Bilateral filtering | Edge-preserving blur |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (training uses mixed precision)
- `transformers` (BERT, DINOv2)
- `datasets` (WikiText-103)
- `mamba-ssm` (selective scan SSM)
- `causal-conv1d` (required by mamba-ssm)

## Citation

```bibtex
@article{shinde2026energy,
  title={Energy-Based Constraint Networks: Learning Structural Coherence Across Modalities},
  author={Shinde, Chirag},
  year={2026},
  url={https://github.com/cs-cmyk/energy-constraint-networks}
}
```

Update with ArXiv ID and Zenodo DOI once available.

## License

MIT
