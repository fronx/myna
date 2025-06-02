"""
Myna model inference utilities
"""

import os
from argparse import Namespace
import torch

from utils import get_n_frames, load_model
from vit import SimpleViT
from audio_utils import sample_spectrogram, get_audio_files, load_raw_audio, get_audio_info, extract_mean_energy, compute_waveform_peaks
import essentia.standard as es
from nnAudio.features.mel import MelSpectrogram


class MynaInference:
    """Myna model inference wrapper"""

    def __init__(self, model_path: str, model_type: str = 'hybrid',
                 hybrid_mode: bool = True, n_samples: int = 50000, sample_rate: int = 16000):
        """
        Initialize Myna inference.

        Args:
            model_path: Path to model checkpoint
            model_type: 'square', 'vertical', or 'hybrid'
            hybrid_mode: Whether to use hybrid mode for hybrid models
            n_samples: Number of samples per embedding chunk
            sample_rate: Target sample rate
        """
        self.model_path = model_path
        self.model_type = model_type
        self.hybrid_mode = hybrid_mode
        self.n_samples = n_samples
        self.sample_rate = sample_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = (128, 2) if model_type == 'vertical' else 16

        # Sanity check
        if hybrid_mode:
            assert model_type == 'hybrid', 'hybrid mode can only be enabled for hybrid model types'

        # Calculate number of frames
        self.n_frames = get_n_frames(
            n_samples=n_samples,
            args=Namespace(
                sr=sample_rate,
                patch_size=self.patch_size
            )
        )

        # Initialize mel spectrogram transform (reused for all segments)
        self.mel_transform = MelSpectrogram(
            sr=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            n_mels=128,
            fmin=60,
            fmax=7800,
            power=2
        )

        self.energy = es.Energy()
        self.model = self._setup_model()

    def _setup_model(self):
        """Setup and load the Myna model"""
        model = SimpleViT(
            image_size=(128, self.n_frames),
            channels=1,
            patch_size=self.patch_size,
            num_classes=50,  # doesn't matter
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=1536,
            additional_patch_size=(128, 2) if self.model_type == 'hybrid' else None
        )

        # Load weights
        load_model(model, self.model_path, self.device, ignore_layers=['linear_head'], verbose=True)
        model.linear_head = torch.nn.Identity()
        model.hybrid_mode = self.hybrid_mode
        model.eval()

        return model

    def _compute_hash_from_samples(self, ms: torch.Tensor) -> str:
        """Compute hash from audio samples only."""
        import hashlib
        samples_bytes = ms.cpu().numpy().tobytes()

        hasher = hashlib.sha256()
        hasher.update(samples_bytes)
        return hasher.hexdigest()


    def _preprocess_audio(self, audio_file: str, profile: bool = False):
        """
        Preprocess audio file by extracting strategic segments and computing spectrograms.

        Args:
            audio_file: Path to audio file
            profile: Whether to output timing information

        Returns:
            tuple: (spectrogram_samples, audio_hash, mean_energy, waveform_peaks, duration_seconds)
        """
        import time

        if profile:
            total_start = time.perf_counter()
            print(f"\nProfiling preprocessing for: {os.path.basename(audio_file)}")

        # Audio loading
        if profile:
            start_time = time.perf_counter()

        # Get audio info without loading full file
        original_sr, total_frames = get_audio_info(audio_file)
        
        # Calculate duration in seconds
        duration_seconds = total_frames / original_sr

        if profile:
            info_time = time.perf_counter() - start_time
            print(f"  Audio info: {info_time:.3f}s (duration: {duration_seconds:.1f}s)")

        # Calculate segment positions and sizes
        segment_positions = [0.15, 0.35, 0.55, 0.75]
        extract_duration_samples = max(self.n_samples, int(0.1 * total_frames))

        if profile:
            mel_start = time.perf_counter()

        segment_spectrograms = []
        audio_segments = []
        for position in segment_positions:
            # Calculate segment boundaries in original sample rate
            start_frame = int(position * total_frames)
            num_frames = min(extract_duration_samples, total_frames - start_frame)

            # Load only this segment
            segment = load_raw_audio(audio_file, self.sample_rate, profile=profile,
                                   start_frame=start_frame, num_frames=num_frames)

            # Store raw audio segment for energy extraction
            audio_segments.append(segment)

            # Convert to mel spectrogram (this will have variable frames)
            segment_ms = self.mel_transform(segment.unsqueeze(0)).squeeze(0)
            segment_ms = segment_ms.unsqueeze(0)  # Shape: (1, n_mels, frames)

            # Use sample_spectrogram to get exactly the right frames (just like original)
            sampled_ms = sample_spectrogram(segment_ms, self.n_frames)
            segment_spectrograms.append(sampled_ms[0])  # Take first (and only) sample: (64, 96)

        if profile:
            mel_time = time.perf_counter() - mel_start
            print(f"  Mel spectrograms (4 segments): {mel_time:.3f}s")

        # Stack all segments: (num_samples, n_mels, n_frames) - same as original
        if profile:
            hash_start = time.perf_counter()

        ms = torch.stack(segment_spectrograms)
        audio_hash = self._compute_hash_from_samples(ms)

        # Extract mean energy from raw audio segments
        mean_energy = extract_mean_energy(audio_segments, self.energy)

        # Compute waveform peaks for visualization
        if profile:
            waveform_start = time.perf_counter()
        
        waveform_peaks = compute_waveform_peaks(audio_file, self.sample_rate)

        if profile:
            waveform_time = time.perf_counter() - waveform_start
            hash_time = time.perf_counter() - hash_start
            total_time = time.perf_counter() - total_start
            print(f"  Hash computation: {hash_time:.3f}s")
            print(f"  Waveform peaks: {waveform_time:.3f}s")
            print(f"  Total preprocessing: {total_time:.3f}s")

        return ms, audio_hash, mean_energy, waveform_peaks, duration_seconds


    def get_audio_files(self, folder_path: str):
        """
        Get list of audio files in folder.

        Args:
            folder_path: Path to folder containing audio files

        Returns:
            list: List of audio file paths

        Raises:
            ValueError: If folder doesn't exist or contains no audio files
        """
        # Validate folder
        if not os.path.isdir(folder_path):
            raise ValueError(f"Error: {folder_path} is not a valid directory")

        # Get audio files
        audio_files = get_audio_files(folder_path)
        if not audio_files:
            raise ValueError(f"No audio files found in {folder_path}")

        return audio_files

    def process_folder(self, folder_path: str, vector_store, progress_callback=None, profile: bool = False, audio_files: list = None, force: bool = False):
        """
        Process all audio files in a folder.

        Args:
            folder_path: Path to folder containing audio files
            vector_store: Vector store for checking existing embeddings and storage
            progress_callback: Optional callback function for progress updates
                             Called with (filename, success, result_or_error)
            profile: Whether to output timing information for preprocessing
            audio_files: Optional list of specific audio files to process (default: all files in folder)
            force: Force recomputation of all tracks, ignoring existing data

        Returns:
            dict: Dictionary mapping filenames to embeddings
        """
        if audio_files is None:
            audio_files = self.get_audio_files(folder_path)
        results = {}

        # Process each audio file
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)

            # Extract strategic segments and compute hash
            ms, audio_hash, mean_energy, waveform_peaks, duration_seconds = self._preprocess_audio(audio_file, profile=profile)

            # Check if file needs reprocessing based on audio content
            if not force and not vector_store.needs_reprocessing(audio_file, audio_hash):
                results[filename] = "skipped"
                if progress_callback:
                    # Get stored energy and waveform for skipped files
                    stored_info = vector_store.get_track_info(audio_file)
                    stored_energy = stored_info.get("energy") if stored_info else None
                    stored_waveform = stored_info.get("waveform") if stored_info else None
                    stored_duration = stored_info.get("duration") if stored_info else duration_seconds
                    progress_callback(filename, True, "skipped", audio_hash, stored_energy, stored_waveform, stored_duration)
                continue

            # Run inference on the already-preprocessed spectrogram
            with torch.no_grad():
                # ms has shape (num_samples, n_mels, n_frames), process each sample individually
                sample_embeds = []
                for i in range(ms.shape[0]):
                    sample_ms = ms[i].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mels, n_frames)
                    embed = self.model(sample_ms)
                    sample_embeds.append(embed)
                embeds = torch.cat(sample_embeds, dim=0)  # Combine all embeddings

            results[filename] = embeds

            if progress_callback:
                progress_callback(filename, True, embeds, audio_hash, mean_energy, waveform_peaks, duration_seconds)

        return results


def create_inference_engine(model_path: str = 'pretrained/myna-hybrid.pth',
                          model_type: str = 'hybrid',
                          hybrid_mode: bool = True):
    """
    Convenience function to create a Myna inference engine with default parameters.

    Args:
        model_path: Path to model checkpoint
        model_type: 'square', 'vertical', or 'hybrid'
        hybrid_mode: Whether to use hybrid mode for hybrid models

    Returns:
        MynaInference: Configured inference engine
    """
    return MynaInference(
        model_path=model_path,
        model_type=model_type,
        hybrid_mode=hybrid_mode,
        n_samples=50000,
        sample_rate=16000
    )