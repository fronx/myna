"""
Convert DJ music collections into *random short mel-spectrogram samples* for Myna training.

The goal is to avoid exporting full-track spectrograms: we sample many short windows per track
(e.g. 3 seconds), convert each window to a mel spectrogram, and save each window as its own file.

This matches Myna-style training where the model sees short excerpts rather than whole tracks.
"""

import os
import pickle
import random
import sqlite3
import tempfile
from pathlib import Path
import argparse

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import librosa
from nnAudio.features.mel import MelSpectrogram
import requests

import hashlib
import json
from typing import List, Tuple, Optional


MYNA_SR = 16000


def download_preview_url(preview_url: str) -> str:
    """Download Apple Music preview to a temp file, return path."""
    with requests.get(preview_url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
            return f.name


def load_tracks_from_musicmapper(db_path: str) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]]]:
    """Load tracks from MusicMapper database.

    Returns:
        (local_tracks, cloud_tracks) where each is a list of (track_id, title, artist, path_or_url) tuples.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Local tracks with file paths
    cursor.execute("""
        SELECT t.track_id, t.title, t.artist, f.file_path
        FROM tracks t
        JOIN files f ON t.track_id = f.track_id
        WHERE f.file_path IS NOT NULL
    """)
    local_tracks = cursor.fetchall()

    # Apple Music tracks without local files (cloud-only)
    cursor.execute("""
        SELECT t.track_id, t.title, t.artist, am.preview_url
        FROM tracks t
        JOIN apple_music_tracks am ON t.track_id = am.track_id
        LEFT JOIN files f ON t.track_id = f.track_id
        WHERE am.preview_url IS NOT NULL AND f.file_path IS NULL
    """)
    cloud_tracks = cursor.fetchall()

    conn.close()
    return local_tracks, cloud_tracks


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
    output_dir: str,
    input_dir: Optional[str] = None,
    musicmapper_db: Optional[str] = None,
    train_split: float = 0.8,
    clip_seconds: float = 3.0,
    samples_per_track: int = 32,
    n_bins: int = 8,
    seed: int = 42
):
    """
    Process DJ collection into train/test splits.

    Args:
        output_dir: Where to save pickle files
        input_dir: Path to music files (flat directory or nested)
        musicmapper_db: Path to MusicMapper SQLite database for Apple Music previews
        train_split: Fraction of tracks to use for training
        clip_seconds: Duration of each sampled window in seconds (default: 3.0)
        samples_per_track: Number of windows to sample per track (default: 32)
        n_bins: Number of coarse time bins for stratified sampling (default: 8)
        seed: Random seed for reproducible sampling (default: 42)
    """
    if not input_dir and not musicmapper_db:
        raise ValueError("At least one of input_dir or musicmapper_db must be provided")

    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    mel_transform = MelSpectrogram(
        sr=MYNA_SR, n_fft=1024, win_length=1024,
        hop_length=160, n_mels=128, fmin=60, fmax=7800, power=2,
        verbose=False
    )

    # Collect tracks from both sources
    # Each track is (track_id, display_name, source_path_or_url, is_preview)
    all_tracks: List[Tuple[str, str, str, bool]] = []

    # Local audio files
    if input_dir:
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.aif', '.aiff'}
        audio_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    audio_files.append(os.path.join(root, file))
        audio_files = sorted(set(audio_files))
        print(f"Found {len(audio_files)} local audio files")
        for f in audio_files:
            tid = stable_track_id(f)
            all_tracks.append((tid, Path(f).stem, f, False))

    # Tracks from MusicMapper database (both local files and Apple Music previews)
    if musicmapper_db:
        if not os.path.exists(musicmapper_db):
            raise FileNotFoundError(f"MusicMapper database not found: {musicmapper_db}")
        db_local, db_cloud = load_tracks_from_musicmapper(musicmapper_db)
        print(f"Found {len(db_local)} local tracks and {len(db_cloud)} Apple Music previews in database")
        for track_id, title, artist, file_path in db_local:
            display = f"{artist} - {title}" if artist else title
            all_tracks.append((track_id, display, file_path, False))
        for track_id, title, artist, preview_url in db_cloud:
            display = f"{artist} - {title}" if artist else title
            all_tracks.append((track_id, display, preview_url, True))

    if not all_tracks:
        print("No tracks found")
        return

    print(f"Total: {len(all_tracks)} tracks")

    random.seed(seed)
    random.shuffle(all_tracks)

    n_train = int(len(all_tracks) * train_split)
    train_tracks = all_tracks[:n_train]
    test_tracks = all_tracks[n_train:]

    print(f"Split: {len(train_tracks)} training, {len(test_tracks)} test")

    manifest_path = os.path.join(output_dir, 'samples.jsonl')
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    def process_tracks(tracks, split_name):
        success_count = 0
        written_samples = 0
        for tid, display_name, source, is_preview in tqdm(tracks, desc=f"Processing {split_name}"):
            temp_file = None
            try:
                # Get audio file path (download if preview)
                if is_preview:
                    temp_file = download_preview_url(source)
                    audio_file = temp_file
                else:
                    audio_file = source

                # Determine duration
                num_frames, sr = get_audio_info(audio_file)
                if num_frames is not None and sr is not None and sr > 0:
                    duration_sec = float(num_frames) / float(sr)
                else:
                    duration_sec = float(librosa.get_duration(filename=audio_file))

                # Hash the track ID to get a stable integer seed (handles UUIDs and hex strings)
                tid_hash = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
                offsets = choose_offsets(
                    duration_sec=duration_sec,
                    clip_sec=clip_seconds,
                    samples_per_track=samples_per_track,
                    n_bins=n_bins,
                    seed=(seed ^ tid_hash) & 0xFFFFFFFF
                )

                safe_stem = display_name.replace('/', '_').replace('\\', '_')[:80]

                for j, start_sec in enumerate(offsets):
                    signal = load_audio_segment(audio_file, start_sec=start_sec, duration_sec=clip_seconds)

                    target_len = int(MYNA_SR * clip_seconds)
                    if signal.shape[-1] < target_len:
                        pad = target_len - signal.shape[-1]
                        signal = torch.nn.functional.pad(signal, (0, pad))
                    elif signal.shape[-1] > target_len:
                        signal = signal[..., :target_len]

                    with torch.no_grad():
                        spec = mel_transform(signal)

                    offset_ms = int(round(start_sec * 1000.0))
                    out_name = f"{safe_stem}__{tid[:16]}__{offset_ms:010d}ms__{j:03d}.pkl"
                    output_file = os.path.join(output_dir, split_name, out_name)

                    with open(output_file, 'wb') as f:
                        pickle.dump(spec.half(), f, protocol=pickle.HIGHEST_PROTOCOL)

                    rec = {
                        'split': split_name,
                        'sample_path': os.path.relpath(output_file, output_dir),
                        'source': source,
                        'is_preview': is_preview,
                        'track_id': tid,
                        'start_sec': float(start_sec),
                        'clip_seconds': float(clip_seconds),
                        'sr': MYNA_SR,
                        'n_mels': 128,
                    }
                    with open(manifest_path, 'a', encoding='utf-8') as mf:
                        mf.write(json.dumps(rec) + "\n")

                    written_samples += 1

                success_count += 1
            except Exception as e:
                tqdm.write(f"Error: {display_name}: {e}")
            finally:
                if temp_file and os.path.exists(temp_file):
                    os.unlink(temp_file)

        tqdm.write(f"{split_name}: wrote {written_samples} samples from {success_count} tracks")
        return success_count

    train_success = process_tracks(train_tracks, "train")
    test_success = process_tracks(test_tracks, "test")

    print(f"\nDone! Train: {train_success}/{len(train_tracks)}, Test: {test_success}/{len(test_tracks)}")
    print(f"Output: {output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare DJ collection for Myna training')
    parser.add_argument('output_dir', help='Where to save processed dataset')
    parser.add_argument('--input-dir', help='Directory containing local music files')
    parser.add_argument('--musicmapper-db', help='Path to MusicMapper SQLite database for Apple Music previews')
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--clip-seconds', type=float, default=3.0, help='Duration of each sampled window in seconds (default: 3.0)')
    parser.add_argument('--samples-per-track', type=int, default=6, help='Number of windows to sample per track (default: 6)')
    parser.add_argument('--n-bins', type=int, default=8, help='Number of coarse time bins for stratified sampling (default: 8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling (default: 42)')
    args = parser.parse_args()

    if not args.input_dir and not args.musicmapper_db:
        parser.error('At least one of --input-dir or --musicmapper-db is required')

    process_dj_collection(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        musicmapper_db=args.musicmapper_db,
        train_split=args.train_split,
        clip_seconds=args.clip_seconds,
        samples_per_track=args.samples_per_track,
        n_bins=args.n_bins,
        seed=args.seed
    )
