"""
Audio processing utilities for Myna model inference
"""

import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from nnAudio.features.mel import MelSpectrogram


def load_raw_audio(filename: str, target_sr: int = 16000):
    """
    Load audio file and return raw audio tensor.

    Args:
        filename: Path to audio file
        target_sr: Target sample rate (default: 16000)

    Returns:
        torch.Tensor: Raw audio tensor of shape (samples,)
    """
    try:
        # Try torchaudio first
        signal, sr = torchaudio.load(filename)
    except RuntimeError:
        # Fallback to librosa for MP3 files
        signal_np, sr = librosa.load(filename, sr=None, mono=False)

        # Convert to torch tensor and ensure proper shape
        if signal_np.ndim == 1:
            signal = torch.from_numpy(signal_np).unsqueeze(0)
        else:
            signal = torch.from_numpy(signal_np)

    # make mono if necessary
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # resample to target sample rate
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        signal = resampler(signal)

    # Return mono audio tensor
    return signal.squeeze(0)


def sample_spectrogram(ms: torch.Tensor, n_frames: int, num_samples: int = 4):
    """
    Sample spectrogram at strategic positions (record store sampling).

    Args:
        ms: Mel spectrogram tensor of shape (1, n_mels, total_frames)
        n_frames: Number of frames per sample
        num_samples: Number of samples to take (default: 4)

    Returns:
        torch.Tensor: Sampled spectrograms of shape (num_samples, n_mels, n_frames)
    """
    # sanity check
    assert ms.dim() == 3 and ms.shape[0] == 1

    total_frames = ms.shape[-1]

    # If track is too short, just take what we can
    if total_frames < n_frames:
        return ms[0].unsqueeze(0)  # Remove batch dim, then add sample dim: (1, n_mels, n_frames)

    # Strategic sampling positions: skip intro/outro, sample middle sections
    # Positions: 15%, 35%, 55%, 75% of track (like sampling a record)
    positions = [0.15, 0.35, 0.55, 0.75][:num_samples]

    samples = []
    for pos in positions:
        start_frame = int(pos * (total_frames - n_frames))
        end_frame = start_frame + n_frames

        # Ensure we don't go out of bounds
        if end_frame <= total_frames:
            sample = ms[0, :, start_frame:end_frame]  # Remove batch dimension: (n_mels, n_frames)
            samples.append(sample)

    # Stack samples if we have any
    if samples:
        return torch.stack(samples)
    else:
        # Fallback: just take the beginning
        return ms[0, :, :n_frames].unsqueeze(0)  # Remove batch dim, then add sample dim


def get_audio_files(folder_path: str):
    """
    Get all audio files from the specified folder.

    Args:
        folder_path: Path to folder containing audio files

    Returns:
        list: Sorted list of audio file paths
    """
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    return sorted(audio_files)