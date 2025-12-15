# Beam Cloud GPU Training

This project uses [Beam Cloud](https://beam.cloud) for GPU-accelerated training since macOS does not support NVIDIA CUDA.

## Quick Start

```bash
# 1. Install Beam
pip install beam-client

# 2. Authenticate (creates ~/.beam/config.ini)
beam config create

# 3. Upload your dataset to a volume
beam cp /path/to/dataset beam://myna-data

# 4. Run training
python beam_train.py --gpu A10G -- \
    --dataroot /volumes/myna-data \
    --task_type contrastive \
    --epochs 100 \
    --checkpoint_dir /volumes/myna-checkpoints/run1
```

## Concepts

### Sandboxes
Sandboxes are isolated cloud environments where code runs. Our `beam_train.py` script creates a sandbox with:
- A GPU-enabled container
- All Python dependencies pre-installed
- Access to persistent volumes for data and checkpoints

Docs: https://docs.beam.cloud/v2/sandbox/overview

### Volumes
Persistent storage that mounts directly to containers. Faster than cloud object storage.

| Command | Description |
|---------|-------------|
| `beam volume list` | List all volumes |
| `beam volume create NAME` | Create a new volume |
| `beam cp LOCAL beam://VOLUME/PATH` | Upload files |
| `beam cp beam://VOLUME/PATH LOCAL` | Download files |
| `beam rm beam://VOLUME/PATH` | Remove files |

This project uses two volumes:
- `myna-data` - Training dataset (mounted at `/volumes/myna-data`)
- `myna-checkpoints` - Model checkpoints (mounted at `/volumes/myna-checkpoints`)

Docs: https://docs.beam.cloud/v2/data/volume

### Container Images
The training environment is defined in `beam_train.py`. Beam automatically handles CUDA setup when a GPU is specified - no custom base image needed:

```python
image = Image(
    python_version="python3.11",
    python_packages="requirements.txt",
)
```

Docs: https://docs.beam.cloud/v2/environment/custom-images

### GPU Options

| GPU | VRAM | Best For |
|-----|------|----------|
| A10G | 24 GB | General training, good price/performance |
| RTX4090 | 24 GB | Similar to A10G |
| H100 | 80 GB | Large models, multi-GPU workloads |

Check availability: `beam machine list`

Docs: https://docs.beam.cloud/v2/environment/gpu

## Usage

### Basic Training
```bash
python beam_train.py -- \
    --dataroot /volumes/myna-data \
    --task_type contrastive \
    --epochs 100 \
    --batch_size 32
```

### With Checkpointing
```bash
python beam_train.py -- \
    --dataroot /volumes/myna-data \
    --task_type contrastive \
    --epochs 100 \
    --checkpoint_dir /volumes/myna-checkpoints/experiment1 \
    --checkpoint_epochs 10
```

### With Weights & Biases
Set your API key in Beam secrets first:
```bash
beam secret create WANDB_API_KEY
```

Then run with wandb flags:
```bash
python beam_train.py -- \
    --dataroot /volumes/myna-data \
    --task_type contrastive \
    --wandb \
    --wandb_project myna \
    --run_name experiment1
```

### Resume Training
```bash
python beam_train.py -- \
    --dataroot /volumes/myna-data \
    --task_type contrastive \
    --resume /volumes/myna-checkpoints/experiment1/model_epoch_50.pth \
    --resume_epochs 50 \
    --epochs 100
```

### Dry Run
Preview the command without running:
```bash
python beam_train.py --dry-run -- --dataroot /volumes/myna-data --task_type contrastive
```

### Select GPU
```bash
python beam_train.py --gpu H100 -- ...
```

## Downloading Results

```bash
# List checkpoint files
beam ls beam://myna-checkpoints/experiment1

# Download a specific checkpoint
beam cp beam://myna-checkpoints/experiment1/model_epoch_100.pth .

# Download entire experiment folder
beam cp -r beam://myna-checkpoints/experiment1 ./checkpoints/
```

## Troubleshooting

### "Volume not found"
Create the volumes first:
```bash
beam volume create myna-data
beam volume create myna-checkpoints
```

### "No GPU available"
Check machine availability:
```bash
beam machine list
```
Try a different GPU type or wait for availability.

### File sync delays
Changes to volumes may take up to 60 seconds to propagate across containers.

### Sandbox timeout
Long-running sandboxes may timeout. For very long training jobs, consider using `@task_queue` decorator instead (see Beam docs).

## Cost Considerations

Beam charges per-second for compute. Sandboxes automatically shut down when the process completes. To minimize costs:
- Use appropriate GPU size (A10G is often sufficient)
- Use checkpointing to resume interrupted runs
- Delete unused volumes: `beam volume delete NAME`

## Links

- Beam Cloud Platform: https://platform.beam.cloud
- Documentation: https://docs.beam.cloud
- Python SDK Reference: https://docs.beam.cloud/v2/reference/py-sdk
