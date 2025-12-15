"""
Convert DJ music collections into *random short mel-spectrogram samples* for Myna training.

The goal is to avoid exporting full-track spectrograms: we sample many short windows per track
(e.g. 3 seconds), convert each window to a mel spectrogram, and save each window as its own file.

This matches Myna-style training where the model sees short excerpts rather than whole tracks.
"""

import os
import pickle
import random
from pathlib import Path
import argparse

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import librosa
from nnAudio.features.mel import MelSpectrogram

import hashlib
import json
from typing import List, Tuple, Optional


MYNA_SR = 16000

def stable_track_id(path: str) -> str:
    """Stable, filesystem-independent ID for a track."""
    h = hashlib.sha1(os.path.abspath(path).encode('utf-8')).hexdigest()
    return h[:16]


def get_audio_info(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (num_frames, sample_rate) if possible, else (None, None)."""
    try:
        info = torchaudio.info(filename)
        return info.num_frames, info.sample_rate
    except Exception:
        return None, None


def load_audio_segment(filename: str, start_sec: float, duration_sec: float) -> torch.Tensor:
    """Load a mono waveform segment, resampled to MYNA_SR. Returns shape [1, T]."""
    # First try torchaudio segmented read (fast, avoids loading whole file)
    num_frames, sr = get_audio_info(filename)
    if sr is not None and num_frames is not None:
        frame_offset = int(max(0.0, start_sec) * sr)
        num_read = int(duration_sec * sr)
        try:
            signal, sr2 = torchaudio.load(filename, frame_offset=frame_offset, num_frames=num_read)
            sr = sr2
        except Exception:
            signal, sr = torchaudio.load(filename)
    else:
        # librosa fallback (may load whole file)
        signal_np, sr = librosa.load(filename, sr=None, mono=False, offset=max(0.0, start_sec), duration=duration_sec)
        if signal_np.ndim == 1:
            signal = torch.from_numpy(signal_np).unsqueeze(0)
        else:
            signal = torch.from_numpy(signal_np)

    # to mono
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # resample
    if sr != MYNA_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=MYNA_SR)
        signal = resampler(signal)

    assert signal.dim() == 2
    return signal


def choose_offsets(duration_sec: float, clip_sec: float, samples_per_track: int, n_bins: int, seed: int) -> List[float]:
    """Choose start offsets (seconds) for windows within a track.

    Uses coarse stratification across time to ensure coverage.
    """
    if duration_sec <= 0:
        return [0.0]

    max_start = max(0.0, duration_sec - clip_sec)
    if max_start <= 0.0:
        return [0.0] * max(1, min(samples_per_track, 4))

    rng = random.Random(seed)
    n_bins = max(1, int(n_bins))
    # If the track is short-ish, reduce bins so bins aren't empty
    n_bins = min(n_bins, max(1, int(duration_sec // clip_sec) + 1))

    offsets: List[float] = []
    # Round-robin bins until we have enough offsets
    while len(offsets) < samples_per_track:
        for b in range(n_bins):
            if len(offsets) >= samples_per_track:
                break
            lo = (b / n_bins) * max_start
            hi = ((b + 1) / n_bins) * max_start
            offsets.append(rng.uniform(lo, hi))
    return offsets


def process_dj_collection(
    input_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    clip_seconds: float = 3.0,
    samples_per_track: int = 32,
    n_bins: int = 8,
    seed: int = 42
):
    """
    Process DJ collection into train/test splits.

    Args:
        input_dir: Path to music files (flat directory or nested)
        output_dir: Where to save pickle files
        train_split: Fraction of tracks to use for training
        clip_seconds: Duration of each sampled window in seconds (default: 3.0)
        samples_per_track: Number of windows to sample per track (default: 32)
        n_bins: Number of coarse time bins for stratified sampling (default: 8)
        seed: Random seed for reproducible sampling (default: 42)
    """
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    mel_transform = MelSpectrogram(
        sr=MYNA_SR, n_fft=1024, win_length=1024,
        hop_length=160, n_mels=128, fmin=60, fmax=7800, power=2,
        verbose=False
    )

    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.aif', '.aiff'}
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    audio_files = sorted(set(audio_files))
    print(f"Found {len(audio_files)} audio files")

    random.seed(seed)
    random.shuffle(audio_files)

    n_train = int(len(audio_files) * train_split)
    train_files = audio_files[:n_train]
    test_files = audio_files[n_train:]

    print(f"Split: {len(train_files)} training, {len(test_files)} test")

    manifest_path = os.path.join(output_dir, 'samples.jsonl')
    # Overwrite on each run
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    def process_files(files, split_name):
        success_tracks = 0
        written_samples = 0
        for audio_file in tqdm(files, desc=f"Processing {split_name}"):
            try:
                # Determine duration without decoding whole file (best-effort)
                num_frames, sr = get_audio_info(audio_file)
                if num_frames is not None and sr is not None and sr > 0:
                    duration_sec = float(num_frames) / float(sr)
                else:
                    # Fallback: decode a bit with librosa to get duration (may still read a lot depending on format)
                    duration_sec = float(librosa.get_duration(path=audio_file))

                tid = stable_track_id(audio_file)
                offsets = choose_offsets(
                    duration_sec=duration_sec,
                    clip_sec=clip_seconds,
                    samples_per_track=samples_per_track,
                    n_bins=n_bins,
                    seed=(seed ^ int(tid, 16)) & 0xFFFFFFFF
                )

                safe_stem = Path(audio_file).stem.replace('/', '_').replace('\\', '_')

                for j, start_sec in enumerate(offsets):
                    signal = load_audio_segment(audio_file, start_sec=start_sec, duration_sec=clip_seconds)

                    # Pad if needed (e.g., end-of-file)
                    target_len = int(MYNA_SR * clip_seconds)
                    if signal.shape[-1] < target_len:
                        pad = target_len - signal.shape[-1]
                        signal = torch.nn.functional.pad(signal, (0, pad))
                    elif signal.shape[-1] > target_len:
                        signal = signal[..., :target_len]

                    with torch.no_grad():
                        spec = mel_transform(signal)

                    # Unique, stable-ish filename per (track, offset index)
                    # Include a coarse millisecond offset for transparency/debugging
                    offset_ms = int(round(start_sec * 1000.0))
                    out_name = f"{safe_stem}__{tid}__{offset_ms:010d}ms__{j:03d}.pkl"
                    output_file = os.path.join(output_dir, split_name, out_name)

                    with open(output_file, 'wb') as f:
                        # Unlabeled dataset: store spectrogram with channel dim (1, n_mels, frames)
                        pickle.dump(spec, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # Append metadata line
                    rec = {
                        'split': split_name,
                        'sample_path': os.path.relpath(output_file, output_dir),
                        'track_path': os.path.abspath(audio_file),
                        'track_id': tid,
                        'start_sec': float(start_sec),
                        'clip_seconds': float(clip_seconds),
                        'sr': MYNA_SR,
                        'n_mels': 128,
                    }
                    with open(manifest_path, 'a', encoding='utf-8') as mf:
                        mf.write(json.dumps(rec) + "\n")

                    written_samples += 1

                success_tracks += 1
            except Exception as e:
                tqdm.write(f"Error: {Path(audio_file).name}: {e}")

        tqdm.write(f"{split_name}: wrote {written_samples} samples from {success_tracks} tracks")
        return success_tracks

    train_success = process_files(train_files, "train")
    test_success = process_files(test_files, "test")

    print(f"\nDone! Train tracks: {train_success}/{len(train_files)}, Test tracks: {test_success}/{len(test_files)}")
    print(f"Output: {output_dir}/")
    print(f"Manifest: {os.path.join(output_dir, 'samples.jsonl')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare DJ collection for Myna training')
    parser.add_argument('input_dir', help='Directory containing music files')
    parser.add_argument('output_dir', help='Where to save processed dataset')
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--clip-seconds', type=float, default=3.0, help='Duration of each sampled window in seconds (default: 3.0)')
    parser.add_argument('--samples-per-track', type=int, default=6, help='Number of windows to sample per track (default: 6)')
    parser.add_argument('--n-bins', type=int, default=8, help='Number of coarse time bins for stratified sampling (default: 8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling (default: 42)')
    args = parser.parse_args()

    process_dj_collection(
        args.input_dir,
        args.output_dir,
        args.train_split,
        args.clip_seconds,
        args.samples_per_track,
        args.n_bins,
        args.seed
    )
