'''
Minimal script example for model inference
'''

import argparse
import os
import glob
from argparse import Namespace
from nnAudio.features.mel import MelSpectrogram
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import librosa
import numpy as np

from utils import get_n_frames, load_model
from vit import SimpleViT
MODEL_PATH = 'pretrained/myna-hybrid.pth' # path to model checkpoint
N_SAMPLES = 50000 # number of samples per embedding
MYNA_SR = 16000 # myna constant


def load_and_preprocess_audio(filename: str):
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
    if sr != MYNA_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=MYNA_SR)
        signal = resampler(signal)

    # sanity check
    assert signal.dim() == 2

    # compute spectrogram
    mel_spec = MelSpectrogram(sr=16000, n_mels=128, verbose=False)
    ms = mel_spec(signal)

    return ms

def get_audio_files(folder_path: str):
    """Get all audio files from the specified folder."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    return sorted(audio_files)

def batch_spectrogram(ms: torch.Tensor, n_frames: int):
    # sanity check
    assert ms.dim() == 3 and ms.shape[0] == 1

    # discard excess frames
    num_chunks = ms.shape[-1] // n_frames
    ms = ms[:, :, :num_chunks * n_frames]

    # split the tensor into chunks and stack them
    chunks = torch.chunk(ms, num_chunks, dim=2)
    batch = torch.stack(chunks)

    return batch


def main():
    parser = argparse.ArgumentParser(description='Process audio files in a folder with Myna model')
    parser.add_argument('folder', help='Path to folder containing audio files')
    parser.add_argument('--model-path', default='pretrained/myna-hybrid.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', default='hybrid', choices=['square', 'vertical', 'hybrid'],
                       help='Model type')
    parser.add_argument('--hybrid-mode', action='store_true', default=True,
                       help='Concatenate embeddings for hybrid models; disable to only use square patches')

    args = parser.parse_args()

    # Validate folder path
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        return

    # Get audio files
    audio_files = get_audio_files(args.folder)
    if not audio_files:
        print(f"No audio files found in {args.folder}")
        return

    print(f"Found {len(audio_files)} audio files in {args.folder}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_size = (128, 2) if args.model_type == 'vertical' else 16

    # number of spectrogram frames to feed into the model
    n_frames = get_n_frames(
        n_samples=N_SAMPLES,
        args=Namespace(
            sr=16000,
            patch_size=patch_size
        )
    )

    # initialize model first
    model = SimpleViT(
        image_size=(128, n_frames),
        channels=1,
        patch_size=patch_size,
        num_classes=50, # doesn't matter
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        additional_patch_size=(128, 2) if args.model_type == 'hybrid' else None
    )

    # now load weights
    load_model(model, args.model_path, device, ignore_layers=['linear_head'], verbose=True)
    model.linear_head = torch.nn.Identity()
    model.hybrid_mode = args.hybrid_mode
    model.eval()

    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # load and preprocess audio
            ms = load_and_preprocess_audio(audio_file)
            ms = batch_spectrogram(ms, n_frames)

            # forward pass
            with torch.no_grad():
                embeds = model(ms)

            tqdm.write(f'✓ {os.path.basename(audio_file)}: embeddings shape {embeds.shape}')

        except Exception as e:
            tqdm.write(f'✗ {os.path.basename(audio_file)}: {e}')

if __name__ == '__main__':
    main()