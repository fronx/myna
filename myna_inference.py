"""
Myna model inference utilities
"""

import os
from argparse import Namespace
import torch

from utils import get_n_frames, load_model
from vit import SimpleViT
from audio_utils import load_and_preprocess_audio, sample_spectrogram, get_audio_files


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
        
        # Initialize and load model
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
    
    def process_audio_file(self, audio_file: str):
        """
        Process a single audio file and return embeddings.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            torch.Tensor: Embeddings tensor
        """
        # Load and preprocess audio
        ms = load_and_preprocess_audio(audio_file, self.sample_rate)
        ms = sample_spectrogram(ms, self.n_frames)
        
        # Forward pass
        with torch.no_grad():
            embeds = self.model(ms)
        
        return embeds
    
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
    
    def process_folder(self, folder_path: str, progress_callback=None):
        """
        Process all audio files in a folder.
        
        Args:
            folder_path: Path to folder containing audio files
            progress_callback: Optional callback function for progress updates
                             Called with (filename, success, result_or_error)
            
        Returns:
            dict: Dictionary mapping filenames to embeddings
        """
        audio_files = self.get_audio_files(folder_path)
        results = {}
        
        # Process each audio file
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            try:
                embeds = self.process_audio_file(audio_file)
                results[filename] = embeds
                
                if progress_callback:
                    progress_callback(filename, True, embeds)
                
            except Exception as e:
                results[filename] = None
                
                if progress_callback:
                    progress_callback(filename, False, str(e))
        
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