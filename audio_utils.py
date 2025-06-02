"""
Audio processing utilities for Myna model inference
"""

import os
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from nnAudio.features.mel import MelSpectrogram
import essentia.standard as es
from statistics import mean


def get_audio_info(filename: str):
    """
    Get audio file metadata without loading the audio.

    Args:
        filename: Path to audio file

    Returns:
        tuple: (sample_rate, num_frames)
    """
    try:
        # Try torchaudio first
        info = torchaudio.info(filename)
        return info.sample_rate, info.num_frames
    except RuntimeError:
        # Fallback to librosa for MP3 files
        import soundfile as sf
        try:
            with sf.SoundFile(filename) as f:
                return f.samplerate, len(f)
        except:
            # Last resort: use librosa (slower but works for more formats)
            y, sr = librosa.load(filename, sr=None, duration=0.1)  # Load tiny sample
            # Estimate total frames
            import os
            file_size = os.path.getsize(filename)
            # Rough estimate based on sample
            estimated_frames = int(file_size / (len(y) * 4) * len(y) / 0.1)
            return sr, estimated_frames


def load_raw_audio(filename: str, target_sr: int = 16000, profile: bool = False,
                   start_frame: int = None, num_frames: int = None):
    """
    Load audio file and return raw audio tensor.

    Args:
        filename: Path to audio file
        target_sr: Target sample rate (default: 16000)
        profile: Whether to output timing information
        start_frame: Optional start frame for partial loading
        num_frames: Optional number of frames to load

    Returns:
        torch.Tensor: Raw audio tensor of shape (samples,)
    """
    import time

    if profile:
        decode_start = time.perf_counter()

    try:
        # Try torchaudio first
        signal, sr = torchaudio.load(filename, frame_offset=start_frame or 0,
                                   num_frames=num_frames or -1)
        if profile:
            decode_time = time.perf_counter() - decode_start
            frames_info = f" ({num_frames} frames)" if num_frames else ""
            print(f"    Torchaudio decode{frames_info}: {decode_time:.3f}s")
    except RuntimeError:
        # Fallback to librosa for MP3 files
        # Note: librosa offset/duration are in seconds, not frames
        # IMPORTANT: When using librosa fallback, we must use the original file's sample rate
        # for offset/duration calculations, not the target sample rate. This was a critical
        # bug that caused "empty tensor" errors during resampling.
        if start_frame is not None or num_frames is not None:
            original_sr, _ = get_audio_info(filename)
            offset_sec = (start_frame / original_sr) if start_frame else 0
            duration_sec = (num_frames / original_sr) if num_frames else None
        else:
            offset_sec = 0
            duration_sec = None

        signal_np, sr = librosa.load(filename, sr=None, mono=False,
                                   offset=offset_sec, duration=duration_sec)
        if profile:
            decode_time = time.perf_counter() - decode_start
            frames_info = f" ({num_frames} frames)" if num_frames else ""
            print(f"    Librosa decode{frames_info}: {decode_time:.3f}s")

        # Convert to torch tensor and ensure proper shape
        if signal_np.ndim == 1:
            signal = torch.from_numpy(signal_np).unsqueeze(0)
        else:
            signal = torch.from_numpy(signal_np)
            
        # Handle empty signals - this typically indicates metadata inconsistencies
        # where the file's reported duration exceeds its actual audio content
        if signal.numel() == 0:
            raise RuntimeError(
                f"Loaded audio segment is empty (start_frame={start_frame}, num_frames={num_frames}). "
                f"This usually indicates the file has metadata that reports longer duration than "
                f"the actual audio content - common with some .m4a and .mp3 files."
            )

    if profile:
        mono_start = time.perf_counter()

    # make mono if necessary
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    if profile:
        mono_time = time.perf_counter() - mono_start
        if mono_time > 0.001:  # Only print if significant
            print(f"    Mono conversion: {mono_time:.3f}s")

    if profile:
        resample_start = time.perf_counter()

    # resample to target sample rate
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        signal = resampler(signal)

        if profile:
            resample_time = time.perf_counter() - resample_start
            print(f"    Resampling {sr}→{target_sr}Hz: {resample_time:.3f}s")

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
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.aiff', '*.aif']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    return sorted(audio_files)


def extract_mean_energy(audio_segments: list, energy_extractor: es.Energy) -> float:
    """Extract mean energy per sample from audio segments using Essentia."""
    normalized_energies = (
        energy_extractor(segment.cpu().numpy().astype('float32')) / len(segment)
        for segment in audio_segments
    )
    return mean(normalized_energies)


def load_audio_segment_with_fallback(filename: str, target_sr: int, start_frame: int, 
                                   num_frames: int, profile: bool = False):
    """
    Load audio segment, returning None if segment is beyond actual file content.
    
    Handles metadata inconsistencies where files report longer duration than actual audio.
    """
    try:
        return load_raw_audio(filename, target_sr, profile=profile,
                            start_frame=start_frame, num_frames=num_frames)
    except RuntimeError as e:
        if "empty" in str(e).lower():
            # This typically indicates the requested segment is beyond the actual
            # audio content due to metadata inconsistencies (common with some .m4a files)
            return None
        else:
            # Re-raise genuine errors (corruption, permission issues, etc.)
            raise


def extract_audio_segments(filename: str, total_frames: int, original_sr: int, target_sr: int,
                         n_samples: int, mel_transform, n_frames: int, profile: bool = False):
    """
    Extract audio segments from strategic positions in the track for robust analysis.
    
    Uses a "record store sampling" approach - taking segments from 15%, 35%, 55%, 
    and 75% positions to avoid intro/outro and capture the song's core content.
    
    Handles files with metadata inconsistencies gracefully by skipping segments 
    that extend beyond the actual audio content.
    
    Args:
        filename: Path to audio file
        total_frames: Total frames reported in metadata
        original_sr: Original sample rate of the file
        target_sr: Target sample rate for processing
        n_samples: Number of samples per embedding chunk
        mel_transform: Mel spectrogram transform function
        n_frames: Number of frames per sample for spectrograms
        profile: Whether to output timing information
        
    Returns:
        tuple: (segment_spectrograms, audio_segments) where both are lists
              containing the successfully extracted segments
    """
    segment_positions = [0.15, 0.35, 0.55, 0.75]
    extract_duration_samples = max(n_samples, int(0.1 * total_frames))
    
    segment_spectrograms = []
    audio_segments = []
    
    for position in segment_positions:
        # Calculate segment boundaries in original sample rate
        start_frame = int(position * total_frames)
        num_frames = min(extract_duration_samples, total_frames - start_frame)
        
        # Skip segments that are too small or go beyond the file
        if num_frames <= 0 or start_frame >= total_frames:
            if profile:
                print(f"    Skipping segment at {position:.1%}: insufficient audio data")
            continue

        # Load segment with robust error handling
        segment = load_audio_segment_with_fallback(
            filename, target_sr, start_frame, num_frames, profile
        )
        
        if segment is None:
            # Segment was beyond actual file content (metadata mismatch)
            if profile:
                print(f"    Skipping segment at {position:.1%}: beyond actual file content")
            continue

        # Store raw audio segment for energy extraction
        audio_segments.append(segment)

        # Convert to mel spectrogram (this will have variable frames)
        segment_ms = mel_transform(segment.unsqueeze(0)).squeeze(0)
        segment_ms = segment_ms.unsqueeze(0)  # Shape: (1, n_mels, frames)

        # Use sample_spectrogram to get exactly the right frames (just like original)
        sampled_ms = sample_spectrogram(segment_ms, n_frames)
        segment_spectrograms.append(sampled_ms[0])  # Take first (and only) sample: (64, 96)
        
    return segment_spectrograms, audio_segments


def compute_waveform_peaks(filename: str, target_sr: int = 16000, samples_per_pixel: int = 512) -> list:
    """
    Compute waveform peaks for WaveSurfer visualization using vectorized NumPy operations.
    
    Args:
        filename: Path to audio file
        target_sr: Target sample rate for processing
        samples_per_pixel: Number of audio samples per waveform pixel
    
    Returns:
        list: Waveform peaks data as list of lists (one per channel)
    """
    # Load full audio file
    signal = load_raw_audio(filename, target_sr)
    
    # Ensure we have shape (channels, samples)
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)  # Add channel dimension
    
    channels = signal.shape[0]
    total_samples = signal.shape[1]
    
    # Calculate number of peaks based on samples per pixel
    num_peaks = total_samples // samples_per_pixel
    if num_peaks == 0:
        num_peaks = 1
    
    peaks_data = []
    
    for channel in range(channels):
        channel_data = signal[channel].numpy()
        
        # Vectorized peak computation using reshape and max
        if total_samples >= samples_per_pixel:
            # Trim to exact multiple of samples_per_pixel for efficient reshaping
            trimmed_length = num_peaks * samples_per_pixel
            trimmed_data = channel_data[:trimmed_length]
            
            # Reshape and compute max absolute value per segment (vectorized)
            reshaped = trimmed_data.reshape(num_peaks, samples_per_pixel)
            peaks = np.max(np.abs(reshaped), axis=1).tolist()
        else:
            # Fallback for very short audio
            peaks = [float(np.max(np.abs(channel_data))) if len(channel_data) > 0 else 0.0]
        
        peaks_data.append(peaks)
    
    return peaks_data
