"""
Beam Cloud GPU Training Script for Myna

Usage:
    python beam_train.py --dataroot data/dj_collection/ [train.py args...]

Data is automatically uploaded to a persistent volume on first run.
"""

import argparse
import os
from typing import Optional

from beam import Image, Volume, Sandbox


# Container image with all dependencies (Beam handles CUDA automatically when GPU is specified)
image = Image(
    python_version="python3.11",
    python_packages="requirements.txt",
)

# Volumes for data and checkpoints
volumes = [
    Volume(name="myna-data", mount_path="/volumes/myna-data"),
    Volume(name="myna-checkpoints", mount_path="/volumes/myna-checkpoints"),
]

# Files to upload to the sandbox
LOCAL_FILES = [
    "train.py",
    "utils.py",
    "vit.py",
]

SANDBOX_FILE = ".sandbox"


def parse_beam_args():
    """Parse arguments for beam_train.py itself (not the training args)."""
    parser = argparse.ArgumentParser(description="Run training on Beam Cloud GPU")
    parser.add_argument("--gpu", type=str, default="RTX4090", choices=["A10G", "RTX4090", "H100"],
                        help="GPU type to use")
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Path to dataset (will be uploaded if local)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the training command without running")
    parser.add_argument("--checkpoint", type=str, default="pretrained/myna-hybrid.pth",
                        help="Path to pretrained checkpoint to resume from")

    # All remaining args are passed to train.py
    args, train_args = parser.parse_known_args()
    return args, train_args


def build_train_command(train_args: list[str], dataroot: str, resume_path: Optional[str] = None) -> str:
    """Build the train.py command from arguments."""
    args_str = " ".join(train_args)
    # Architecture is auto-inferred from checkpoint when --resume is used
    cmd = f"cd /workspace && python train.py --dataroot {dataroot} --task_type contrastive --batch_size 256 --learning_rate 1e-5 --train_only_head_epochs 5"
    if resume_path:
        cmd += f" --resume {resume_path}"
    # Save checkpoints to volume, every N epochs
    cmd += " --checkpoint_dir /volumes/myna-checkpoints --checkpoint_epochs 50"
    if args_str:
        cmd += f" {args_str}"
    return cmd


def upload_checkpoint_if_needed(local_checkpoint: str):
    """Upload pretrained checkpoint to volume if not already present."""
    import subprocess

    if not local_checkpoint or not os.path.exists(local_checkpoint):
        return None

    filename = os.path.basename(local_checkpoint)
    volume_path = f"/volumes/myna-checkpoints/{filename}"

    # Check if already uploaded (list directory and look for filename)
    result = subprocess.run(["beam", "ls", "myna-checkpoints/"], capture_output=True, text=True)
    if result.returncode == 0 and filename in result.stdout:
        print(f"Checkpoint already on volume: {volume_path}")
        return volume_path

    print(f"Uploading checkpoint {local_checkpoint}...")
    result = subprocess.run(["beam", "cp", local_checkpoint, f"beam://myna-checkpoints/{filename}"])
    if result.returncode != 0:
        raise RuntimeError("Checkpoint upload failed")
    print(f"Checkpoint uploaded to {volume_path}")
    return volume_path


def upload_archive_if_needed(sb, local_dataroot: str):
    """Upload dataset archive to volume if not already present."""
    import subprocess
    import tarfile

    volume_tar = "/volumes/myna-data/data.tar.gz"

    # Check if archive exists on volume
    check = sb.process.run_code(f"""
import os
print(f"archive:{{os.path.exists('{volume_tar}')}}")
""")
    if "archive:True" in check.result:
        print("Archive already on volume")
        return

    # Create and upload archive
    local_tar = os.path.join(local_dataroot, "data.tar.gz")
    if not os.path.exists(local_tar):
        print(f"Packaging data from {local_dataroot}...")
        with tarfile.open(local_tar, "w:gz") as tar:
            tar.add(local_dataroot, arcname=".", filter=lambda t: None if t.name.endswith(".tar.gz") else t)
    else:
        print(f"Using existing archive: {local_tar}")

    tar_size_mb = os.path.getsize(local_tar) / (1024 * 1024)
    print(f"Uploading {tar_size_mb:.1f} MB archive via beam cp...")

    result = subprocess.run(["beam", "cp", local_tar, "beam://myna-data/data.tar.gz"])
    if result.returncode != 0:
        raise RuntimeError("beam cp failed")
    print("Archive uploaded")


def extract_to_local(sb, local_data_path: str):
    """Extract archive from volume to local disk (fast). Returns path to use for training."""
    volume_tar = "/volumes/myna-data/data.tar.gz"

    # Check if already extracted
    check = sb.process.run_code(f"""
import os
train_dir = "{local_data_path}/train"
count = len(os.listdir(train_dir)) if os.path.isdir(train_dir) else 0
print(f"count:{{count}}")
""")
    if "count:0" not in check.result:
        print(f"Data already extracted at {local_data_path}")
        return

    print(f"Extracting archive to local disk ({local_data_path})...")
    extract_proc = sb.process.exec("bash", "-c", f"mkdir -p {local_data_path} && tar -xzf {volume_tar} -C {local_data_path}")
    for line in extract_proc.logs:
        print(line, end="")
    exit_code = extract_proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"Extraction failed with exit code {exit_code}")
    print("Extraction complete")


def run_training(gpu: str, dataroot: str, checkpoint: str, train_args: list[str], dry_run: bool = False):
    """Create sandbox and run training."""

    # Use local disk for training data (fast), volume only stores the archive
    local_data_path = "/tmp/myna-data"

    # Upload checkpoint to volume and get the volume path
    resume_path = upload_checkpoint_if_needed(checkpoint)
    train_cmd = build_train_command(train_args, local_data_path, resume_path)

    if dry_run:
        print("=== DRY RUN ===")
        print(f"GPU: {gpu}")
        print(f"Checkpoint: {checkpoint} -> {resume_path}")
        print(f"Command: {train_cmd}")
        print(f"Files to upload: {LOCAL_FILES}")
        return

    # Check for existing sandbox
    sandbox_id = None
    if os.path.exists(SANDBOX_FILE):
        with open(SANDBOX_FILE) as f:
            sandbox_id = f.read().strip() or None

    if sandbox_id:
        print(f"Connecting to existing sandbox: {sandbox_id}")
        sb = Sandbox().connect(sandbox_id)
    else:
        print(f"Creating Beam sandbox with {gpu} GPU...")
        sandbox = Sandbox(
            name="myna-training",
            image=image,
            gpu=gpu,
            volumes=volumes,
            keep_warm_seconds=-1,
        )
        sb = sandbox.create()
        sandbox_id = sb.sandbox_id()
        with open(SANDBOX_FILE, "w") as f:
            f.write(sandbox_id)
        print(f"Sandbox created: {sandbox_id}")

    # Upload training files
    print("Uploading training files...")
    for local_file in LOCAL_FILES:
        if os.path.exists(local_file):
            remote_path = f"/workspace/{local_file}"
            sb.fs.upload_file(local_file, remote_path)
            print(f"  Uploaded {local_file} -> {remote_path}")
        else:
            print(f"  Warning: {local_file} not found, skipping")

    # Upload archive to volume, then extract to local disk
    upload_archive_if_needed(sb, dataroot)
    extract_to_local(sb, local_data_path)

    # Check GPU availability
    print("\nVerifying GPU...")
    gpu_check = sb.process.run_code("""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")
    print(gpu_check.result)

    # Run training (non-blocking to stream output)
    print(f"\nStarting training...")
    print(f"Command: {train_cmd}")
    print("=" * 60)

    process = sb.process.exec("bash", "-c", train_cmd)

    # Stream output in real-time with reconnection on connection drops
    import time
    from grpc import RpcError
    max_retries = 5
    retry_count = 0

    while True:
        try:
            for line in process.logs:
                print(line, end="")
                retry_count = 0  # Reset on successful read
            break  # Normal completion
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Killing process...")
            process.kill()
            break
        except RpcError as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\n\nConnection lost after {max_retries} retries. Training continues on remote.")
                print("Rerun this script to reconnect.")
                return
            wait_time = min(2 ** retry_count, 30)
            print(f"\n[Connection lost: {e.code()}. Reconnecting in {wait_time}s... ({retry_count}/{max_retries})]")
            time.sleep(wait_time)
            # Reconnect to sandbox
            sb = Sandbox().connect(sandbox_id)
            # Get process status - if still running, continue streaming
            exit_code, _ = process.status()
            if exit_code >= 0:
                print(f"\n[Process already finished with exit code {exit_code}]")
                break

    # Wait for completion and get exit code
    exit_code = process.wait()
    print("=" * 60)
    print(f"Training finished with exit code: {exit_code}")

    print("\nCheckpoints saved to volume 'myna-checkpoints'")
    print("Download with: beam cp beam://myna-checkpoints/model_epoch_N.pth .")
    print("Done. Sandbox still running - rerun to continue training.")


def main():
    args, train_args = parse_beam_args()
    run_training(
        gpu=args.gpu,
        dataroot=args.dataroot,
        checkpoint=args.checkpoint,
        train_args=train_args,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
