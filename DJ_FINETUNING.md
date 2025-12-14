# Contrastive Fine-tuning for DJ Music Collections

## Executive Summary

This document outlines how to fine-tune the Myna model on DJ music collections using **masking-based contrastive learning** to improve embedding semantics for DJ-specific use cases like track similarity, set construction, and harmonic mixing.

**Key Insight from Myna Paper ([arXiv:2502.12511](https://arxiv.org/html/2502.12511v2)):**

Positive pairs should be **two masked segments from the same track** (90% token masking), NOT different tracks from the same context. This approach:
- Preserves musical properties (key, BPM, timbre) that traditional augmentations destroy
- Forces the model to learn robust representations from limited visible context
- Outperforms traditional augmentation strategies (pitch shift, time stretch)
- Requires no special data organization - just audio files

## What We Have Available

### Model & Checkpoints
- **Pretrained Myna-Hybrid model** ([pretrained/myna-hybrid.pth](pretrained/myna-hybrid.pth), 117MB)
  - Vision Transformer encoder with 12 layers, 384-dim embeddings
  - Hybrid patch processing (square 16x16 + vertical 128x2)
  - Concatenated embeddings: 768 dimensions total
  - Pretrained on large-scale music dataset

### Training Infrastructure
- **Contrastive training pipeline** ([train.py](train.py))
  - SogCLR loss implementation ([utils.py:569-579](utils.py#L569-L579))
  - Multi-GPU distributed training support
  - Checkpoint saving/resuming
  - WandB experiment tracking

- **Audio processing** ([audio_utils.py](audio_utils.py), [myna_inference.py](myna_inference.py))
  - Supports: MP3, WAV, FLAC, M4A, AAC, OGG
  - Mel spectrogram computation (128 mels, 16kHz)
  - Strategic audio sampling (4 segments per track)

- **Data loading** ([utils.py:500-523](utils.py#L500-L523))
  - MelSpectrogramDataset for pickle files
  - Unlabeled dataset support (`--unlabeled` flag)
  - Multi-view augmentation for contrastive pairs

### Evaluation Tools
- **Embedding evaluation** ([evaluate.py](evaluate.py))
  - Pre-compute embeddings for test sets
  - Grid search over classifier hyperparameters
  - Multiple evaluation metrics

## What We CAN Do

### ✅ Myna Contrastive Learning Methodology

Based on the Myna paper ([arXiv:2502.12511](https://arxiv.org/html/2502.12511v2)), the correct approach uses:

**Positive Pair Construction:**
- Two segments from the **same track** (3-second segments)
- Each segment converted to mel spectrogram
- **90% token masking** applied to each view independently
- No traditional augmentations (no pitch shift, time stretch, etc.)

**Why This Works:**
1. **Preserves musical properties**: Key, BPM, timbre remain intact
2. **Forces robust representations**: Model learns to infer from 10% visible context
3. **Musically meaningful**: Unlike pitch shifting, masking doesn't destroy harmonic relationships
4. **Simple and effective**: Single augmentation strategy, proven to work

**What the Model Learns:**
- Track-level identity and coherence
- Robust to missing/incomplete information
- Semantic music features that generalize
- Temporal structure within tracks

**Requirements:**
- Only audio files (no labels, no organization needed)
- No metadata required
- Works with any music collection

### ✅ Training Capabilities

- Start from pretrained weights (transfer learning)
- Freeze backbone, train only projection head
- Progressive unfreezing (head first, then full model)
- Learning rate scheduling (cosine, warmup)
- Gradient clipping for stability
- Multi-GPU distributed training

### ✅ Evaluation Capabilities

- Embedding space visualization (t-SNE, UMAP)
- Nearest neighbor retrieval quality
- Track similarity ranking
- Metadata-based clustering metrics (key, BPM, genre)
- Key compatibility accuracy (if labeled)
- Before/after comparison with pretrained model

## What We CAN'T Do (Without Additional Work)

### ❌ Out of Scope

**1. Raw Audio to Pickle Preprocessing**
- No script to convert DJ collections → pickle format
- Need to build: `prepare_dj_dataset.py`

**2. Automatic Positive Pair Mining**
- No script to discover similar tracks automatically
- Need to build: Manual curation or heuristic-based pairing

**3. Real-time Inference API**
- Current inference is batch-only
- Need to build: REST API or streaming inference

**4. DJ Set Sequence Modeling**
- Model treats tracks independently
- No sequential/temporal modeling across tracks in a set
- Would need: RNN/Transformer over track embeddings

**5. Multi-modal Learning**
- No support for combining audio + metadata + user behavior
- Purely audio-based

## How We Will Know the Result is Improved

### Quantitative Metrics

**1. Retrieval Quality Metrics**

**Nearest Neighbor Quality**
- Query: Any track from your collection
- Expected: Musically similar tracks appear in top-K results
- Metric: Human judgment (listening test) or metadata correlation
- Target: Noticeably better musical coherence than pretrained model

**Key Compatibility Retrieval** (if key labels available)
- Query: Track in key X
- Expected: Compatible keys ranked higher than incompatible
- Metric: Mean Reciprocal Rank (MRR) for compatible keys
- Target: MRR > 0.5 for Camelot wheel neighbors

**BPM Similarity Ranking** (if BPM available)
- Query: Track at X BPM
- Expected: Tracks within ±5 BPM ranked higher
- Metric: Normalized Discounted Cumulative Gain (NDCG@20)
- Target: NDCG > 0.7

**2. Embedding Space Quality**

**Metadata-Based Clustering** (if labels available)
- Calculate: Do tracks with same key/genre/BPM cluster together?
- Metric: Silhouette score for any available categorical labels
- Target: Higher silhouette score than pretrained model

**Key Cluster Coherence** (if key labels available)
- Calculate: Silhouette score for key-based clusters
- Target: Silhouette score > 0.3 (vs ~0.1 for pretrained)

**3. Downstream Task Performance**

If you have labeled data for any task:
- Genre classification accuracy
- Key detection accuracy
- BPM range classification
- Energy level prediction

Compare fine-tuned vs pretrained embeddings as features for these tasks.

### Qualitative Evaluation

**Manual Inspection**
- Select 10-20 "query" tracks from your collection
- Get top-10 nearest neighbors before/after fine-tuning
- Human judgment: Are results more relevant for DJing?

**DJ Use Case Testing**
- "Find tracks that mix well after this one"
- "Find tracks with similar energy"
- "Find harmonic matches"
- Count: How many of top-10 results are actually usable?

**Embedding Visualization**
- t-SNE or UMAP plot colored by any available metadata:
  - Key (if available)
  - BPM range
  - Genre
- Visual inspection: Are similar tracks clustered?

### A/B Testing Framework

```python
# Pseudocode for evaluation
def evaluate_model(model, test_queries, metadata):
    results = {}

    # 1. Metadata-based clustering (if labels available)
    if metadata.get('keys'):
        embeddings = [model.get_embedding(t) for t in test_queries]
        results['key_silhouette'] = silhouette_score(embeddings, metadata['keys'])

    if metadata.get('genres'):
        results['genre_silhouette'] = silhouette_score(embeddings, metadata['genres'])

    # 2. Key compatibility retrieval (if key labels available)
    if metadata.get('keys'):
        for query_track in test_queries:
            neighbors = model.get_nearest_neighbors(query_track, k=20)
            mrr = compute_mrr_for_compatible_keys(query_track, neighbors, metadata['keys'])
            results['key_mrr'].append(mrr)

    return results

# Compare pretrained vs fine-tuned
pretrained_results = evaluate_model(pretrained_model, test_queries, metadata)
finetuned_results = evaluate_model(finetuned_model, test_queries, metadata)

print_comparison(pretrained_results, finetuned_results)
```

## Implementation Plan

### Phase 1: Data Preparation

**Step 1.1: Organize Your DJ Collection**

For **training**, you simply need audio files - no special organization required:

```bash
dj_collections/
├── all_tracks/              # All your music files
│   ├── track001.mp3
│   ├── track002.mp3
│   ├── track003.mp3
│   └── ...
```

For **evaluation** (optional but helpful), keep metadata if you have it:

```bash
dj_collections/
├── all_tracks/              # All your music files
│   ├── track001.mp3
│   ├── track002.mp3
│   └── ...
└── metadata/                # Optional - helps measure quality
    ├── keys.json            # {"track001.mp3": "1A", ...}
    ├── bpms.json            # {"track001.mp3": 128, ...}
    └── genres.json          # {"track001.mp3": "house", ...}
```

**Positive Pairs (Training):**
- Two segments from the **same track** (90% masked)
- Negatives: segments from different tracks in the batch

**Step 1.2: Create Preprocessing Script**

Create `prepare_dj_dataset.py` (using existing infrastructure):

```python
"""
Convert DJ music collections to pickle format for Myna training.
Uses existing audio_utils and myna_inference infrastructure.
"""

import os
import pickle
import random
from pathlib import Path
import torch
from tqdm import tqdm

# Use existing infrastructure
from audio_utils import load_raw_audio, get_audio_files
from nnAudio.features.mel import MelSpectrogram

def process_dj_collection(input_dir, output_dir, train_split=0.8):
    """
    Process DJ collection into train/test splits.

    Args:
        input_dir: Path to music files (flat directory or nested)
        output_dir: Where to save pickle files
        train_split: Fraction of tracks to use for training
    """
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    # Use existing mel spectrogram config from myna_inference.py
    mel_transform = MelSpectrogram(
        sr=16000, n_fft=1024, win_length=1024,
        hop_length=160, n_mels=128, fmin=60, fmax=7800, power=2
    )

    # Get all audio files using existing function
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for ext in ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg']:
            for file in files:
                if file.lower().endswith(f'.{ext}'):
                    audio_files.append(os.path.join(root, file))

    audio_files = sorted(set(audio_files))
    random.seed(42)
    random.shuffle(audio_files)

    # Split at track level
    n_train = int(len(audio_files) * train_split)
    train_files = audio_files[:n_train]
    test_files = audio_files[n_train:]

    print(f"Processing {len(train_files)} training tracks, {len(test_files)} test tracks")

    def process_files(files, split_name):
        for audio_file in tqdm(files, desc=f"Processing {split_name}"):
            try:
                # Use existing audio loading infrastructure
                audio = load_raw_audio(audio_file, target_sr=16000)

                # Compute mel spectrogram
                with torch.no_grad():
                    spec = mel_transform(audio.unsqueeze(0))

                # Save as pickle (unlabeled format: just spec, no label)
                output_file = f"{output_dir}/{split_name}/{Path(audio_file).stem}.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(spec.squeeze(0), f)  # Shape: (1, n_mels, n_frames)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    process_files(train_files, "train")
    process_files(test_files, "test")

    print("✓ Dataset preparation complete!")
    print(f"  Train: {output_dir}/train/")
    print(f"  Test: {output_dir}/test/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory containing music files')
    parser.add_argument('output_dir', help='Where to save processed dataset')
    parser.add_argument('--train-split', type=float, default=0.8)
    args = parser.parse_args()

    process_dj_collection(args.input_dir, args.output_dir, args.train_split)
```

**Step 1.3: Run Preprocessing**

```bash
source venv/bin/activate
python prepare_dj_dataset.py /path/to/dj_collections /path/to/processed_data
```

### Phase 2: Baseline Evaluation

Before fine-tuning, establish baseline metrics using the pretrained model. This gives us a reference point to measure improvement.

**Step 2.1: Generate Baseline Embeddings**

Using the pretrained `myna-hybrid.pth` checkpoint, compute embeddings for your entire test set. For each track:
1. Load the spectrogram from the pickle file
2. Pass it through the model (without the classification head)
3. Store the resulting 768-dimensional embedding vector

Save these embeddings in a structured format (e.g., `{filename: embedding_vector}`) for later comparison.

**Step 2.2: Compute Baseline Metrics**

Since training is purely track-level contrastive learning (no labels), evaluation focuses on embedding quality:

1. **Nearest neighbor inspection**: For a sample of tracks, retrieve top-10 neighbors and manually assess whether they sound musically similar.

2. **Metadata correlation** (if available): If you have key, BPM, or genre labels for some tracks, measure whether the embedding space respects these properties (e.g., do tracks in the same key cluster together?).

3. **Visualization**: Generate a t-SNE or UMAP plot of the embeddings. Look for meaningful structure - tracks should cluster by sonic similarity, not randomly scatter.

4. **Downstream task probe** (optional): Train a simple classifier on the embeddings for any labeled task you have (genre, key, mood). This measures how much useful information the embeddings encode.

Record baseline numbers for any metrics you can compute.

### Phase 3: Contrastive Fine-tuning

**Step 3.1: Training Configuration**

Create `configs/dj_contrastive.sh`:

```bash
#!/bin/bash

# DJ Collection Contrastive Fine-tuning (Myna methodology)
python train.py \
    --dataroot /path/to/processed_data \
    --task_type contrastive \
    --unlabeled \
    --mask_ratio 0.9 \
    --max_frame_distance 32 \
    \
    --resume pretrained/myna-hybrid.pth \
    --ignore_layers linear_head \
    \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 50 \
    \
    --optimizer adam \
    --learning_rate 1e-4 \
    --lr_schedule cosine \
    --warmup_epochs 5 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    \
    --sogclr_tau 0.1 \
    --sogclr_gamma 0.9 \
    --gamma_schedule cosine \
    \
    --train_only_head_epochs 5 \
    \
    --checkpoint_dir checkpoints/dj_finetuned \
    --checkpoint_epochs 5 \
    \
    --wandb \
    --wandb_project myna-dj-finetuning \
    --run_name dj_contrastive_v1 \
    \
    --seed 42
```

**Key Parameter Explanations:**

- `--task_type contrastive --unlabeled`: Contrastive learning without labels
- `--mask_ratio 0.9`: **90% token masking** as per Myna paper - this is the core augmentation strategy
- `--max_frame_distance 32`: Positive pairs sampled close in time (within same track)
- `--resume pretrained/myna-hybrid.pth`: Start from pretrained weights (transfer learning)
- `--ignore_layers linear_head`: Don't load classification head (we don't need it)
- `--train_only_head_epochs 5`: First 5 epochs, only train projection head (freeze encoder)
- `--learning_rate 1e-4`: Lower LR for fine-tuning (vs 3e-4 for training from scratch)
- `--sogclr_tau 0.1`: Temperature for contrastive loss (lower = harder negatives)
- `--sogclr_gamma 0.9`: Momentum for updating running statistics

**Why 90% Masking?**
- Preserves musical properties (key, BPM, timbre) unlike pitch/time augmentations
- Forces model to learn robust representations from limited context
- Paper shows this consistently outperforms lower masking ratios
- No need for traditional data augmentation

**Step 3.2: Run Training**

```bash
source venv/bin/activate
chmod +x configs/dj_contrastive.sh
./configs/dj_contrastive.sh
```

Monitor training:
- Loss should decrease steadily
- Watch WandB dashboard for metrics
- Training time: ~2-5 hours on modern GPU (depends on dataset size)

**Step 3.3: Multi-GPU Training (Optional)**

If you have multiple GPUs:

```bash
torchrun --nproc_per_node=4 train.py \
    [same arguments as above] \
    --dist_backend nccl
```

### Phase 4: Evaluation & Comparison

**Step 4.1: Generate Fine-tuned Embeddings**

After training completes, generate embeddings using your best checkpoint (typically the one with lowest training loss, found in `checkpoints/dj_finetuned/`).

Use the exact same process as Phase 2, but with the fine-tuned checkpoint instead of `pretrained/myna-hybrid.pth`. This ensures a fair comparison - same test data, same embedding extraction method, only the model weights differ.

Save these embeddings separately (e.g., `embeddings/finetuned_test.pkl`) to compare against the baseline.

**Step 4.2: Run Comparative Evaluation**

Run the same evaluation from Phase 2 on the fine-tuned embeddings and compare:

1. **Side-by-side neighbor comparison**: For the same query tracks, compare top-10 neighbors from pretrained vs fine-tuned. Which returns more musically coherent results?

2. **Metadata correlation** (if available): Did clustering by key/BPM/genre improve? Compute silhouette scores or similar metrics if you have labels.

3. **Downstream task probe**: If you trained classifiers on baseline embeddings, retrain on fine-tuned embeddings. Did accuracy improve?

4. **Visual comparison**: Generate t-SNE plots for both embedding sets. Does the fine-tuned version show tighter, more meaningful clusters?

Key question: Do the embeddings better capture the musical properties relevant to your DJ workflow?

If metrics decreased or stayed flat, see the "Risk Mitigation" section for troubleshooting.

**Step 4.3: Qualitative Testing**

Numbers tell part of the story, but DJ workflow is ultimately about musical intuition. Perform manual evaluation:

**Listening test:**
1. Select 10-20 "query" tracks you know well from your collection
2. For each query, retrieve the top-10 nearest neighbors using both pretrained and fine-tuned embeddings
3. Listen to the results and judge: Which model returns tracks you'd actually consider mixing with the query?

**Visual inspection:**
Generate t-SNE/UMAP plots for both embedding sets, colored by any metadata you have:
- Key (if labeled - do harmonically compatible tracks group?)
- BPM range (do similar tempos cluster?)
- Genre/style (subjective but informative)

Compare the plots: Does the fine-tuned model show tighter, more meaningful clusters?

**DJ workflow test:**
Pick a track and use the fine-tuned model to find candidates for:
- "What should I play next?" (overall similarity)
- "What mixes well harmonically?" (if key-aware)
- "What has similar energy?" (subjective but important)

Document specific examples where fine-tuning helped or hurt - these inform whether to iterate further.


### Phase 5: Iteration & Refinement

**Hyperparameter Tuning**

If results aren't satisfactory, try adjusting:

1. **Temperature (`--sogclr_tau`)**
   - Lower (0.05): Harder contrastive task, more discrimination
   - Higher (0.2): Easier task, more generalization

2. **Learning Rate**
   - Too high: Unstable, forgets pretrained knowledge
   - Too low: Slow adaptation, minimal improvement
   - Try: 5e-5, 1e-4, 3e-4

3. **Training Duration**
   - More epochs if still improving
   - Early stopping if overfitting (test loss increases)

4. **Masking Ratio**
   - Paper recommends 90%, but you can experiment
   - Try: 0.75, 0.85, 0.9, 0.95
   - Higher masking = harder task, potentially better representations

5. **Temporal Distance**
   - `--max_frame_distance` controls how far apart positive pairs can be
   - Smaller (16): Forces model to learn fine-grained temporal coherence
   - Larger (64): More diversity between positive pairs
   - Default (32): Good balance

6. **Freezing Strategy**
   - Try `--train_only_head_epochs 10` (more head-only training)
   - Try `--unfreeze_last_n_layers 3` (only fine-tune top layers)

## Expected Outcomes

### Success Criteria

**Minimum Viable Success:**
- Qualitative inspection: 7/10 top nearest neighbors are DJ-appropriate
- Visually tighter clusters in t-SNE compared to pretrained

**Good Success:**
- Noticeably better nearest neighbors in listening tests
- Improved silhouette scores for key/genre clusters (if labels available)
- Key-compatible tracks rank in top-20 (if keys available)

**Excellent Success:**
- Consistently better results across all query tracks
- Clear visual clustering by key/BPM/genre in t-SNE plots
- Could build production DJ recommendation system
- Embeddings encode musically meaningful similarity

### Timeline Estimate

- **Week 1**: Data preparation, baseline evaluation
- **Week 2**: Initial fine-tuning experiments
- **Week 3**: Hyperparameter tuning, iteration
- **Week 4**: Final evaluation, documentation

### Resource Requirements

**Computational:**
- GPU: RTX 3090 or better (24GB VRAM recommended)
- Training time: 2-5 hours per experiment
- Storage: ~1-5GB for processed dataset (depends on collection size)

**Data:**
- Minimum: 1000 tracks (no organization required)
- Recommended: 5000+ tracks for good generalization
- For evaluation: Having metadata (key, BPM, genre) for some tracks helps measure quality
- More diverse music = better learned representations

## Risk Mitigation

### Potential Issues & Solutions

**Issue 1: Overfitting to Specific Style**
- **Symptom**: Great on familiar tracks, poor on general music
- **Solution**: Include diverse tracks (different genres/styles)
- **Solution**: Use lower learning rate, more regularization

**Issue 2: Forgetting Pretrained Knowledge**
- **Symptom**: Fine-tuned model worse than pretrained
- **Solution**: Lower learning rate (1e-5 instead of 1e-4)
- **Solution**: More aggressive freezing (only train head + last 2 layers)
- **Solution**: Shorter training (10-20 epochs instead of 50)

**Issue 3: No Improvement**
- **Symptom**: Metrics stay flat
- **Solution**: Check data quality (are audio files intact? diverse enough?)
- **Solution**: Verify masking is working correctly
- **Solution**: Visualize embeddings to diagnose what's wrong

**Issue 4: Training Instability**
- **Symptom**: Loss spikes, NaN values
- **Solution**: Lower learning rate
- **Solution**: Enable gradient clipping (`--grad_clip 1.0`)
- **Solution**: Reduce batch size

## Next Steps After Fine-tuning

Once you have a fine-tuned model with improved embeddings:

1. **Build DJ Tools**
   - Track recommendation API
   - Harmonic mixing assistant
   - Set analysis and visualization

2. **Expand Dataset**
   - Add more tracks from different genres/sources
   - Include radio shows, mixes from SoundCloud/Mixcloud
   - Continuous fine-tuning as collection grows

3. **Multi-Task Learning**
   - Combine with key detection, BPM prediction
   - Joint training for better representations

4. **Production Deployment**
   - API for real-time track similarity
   - Integration with DJ software (Traktor, Rekordbox, etc.)
   - Mobile app for crate digging

## References & Resources

**In This Codebase:**
- Training: [train.py](train.py)
- Data loading: [utils.py](utils.py) (MelSpectrogramDataset)
- Contrastive loss: [utils.py](utils.py) (GCLoss_v1)
- Audio processing: [audio_utils.py](audio_utils.py)
- Inference: [myna_inference.py](myna_inference.py)

**External Resources:**
- **Myna paper**: https://arxiv.org/html/2502.12511v2 (masking-based contrastive learning)
- SogCLR paper: https://arxiv.org/abs/2202.12387 (contrastive loss used in Myna)
- SimCLR paper: https://arxiv.org/abs/2002.05709 (foundational contrastive learning)
- Music similarity surveys: https://arxiv.org/abs/1906.10508
- Camelot Wheel (key mixing): https://mixedinkey.com/camelot-wheel/

**Key Extraction Tools (if needed):**
- Essentia: `essentia.standard.KeyExtractor()`
- librosa: `librosa.key_to_degrees()`
- Keyfinder: https://github.com/mikebrady/keyfinder-cli

## Conclusion

This codebase provides all the core infrastructure needed for masking-based contrastive fine-tuning on DJ collections. The main work is:

1. **Data prep**: Convert audio → pickles (1-2 days) - no special organization needed
2. **Baseline**: Evaluate pretrained model (1 day)
3. **Fine-tune**: Run training with 90% masking (1-2 weeks)
4. **Evaluate**: Compare and iterate (ongoing)

**Key Advantages of Myna's Approach:**
- **Simple**: Just audio files, no labels or metadata required
- **Effective**: 90% masking proven to outperform traditional augmentations
- **Musically sound**: Preserves harmonic and rhythmic properties
- **Flexible**: Works with any music collection

The pretrained Myna-Hybrid model gives you a strong starting point. By fine-tuning on your DJ collection using the same masking-based contrastive approach, the model will adapt to:
- Your specific music genres and styles
- The sonic characteristics of your collection
- Implicit similarity relationships in your music

This should yield substantial improvements in embedding quality for DJ use cases like track similarity, harmonic mixing, and set construction.

Ready to start? Begin with Phase 1: Data Preparation.
