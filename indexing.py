"""
MusicMapper Indexing Library

Reusable functions for audio processing, embedding generation,
and QDrant operations. Extracted from index.py to enable
background worker and other scripts.
"""

from sklearn.decomposition import PCA
from typing import Dict, List
import torch
import numpy as np
from myna_inference import MynaInference
from vector_store import MynaVectorStore
from qdrant_utils import is_qdrant_running, wait_for_qdrant
from qdrant_client.models import PointStruct


def connect_to_qdrant(qdrant_url: str) -> MynaVectorStore:
    """
    Connect to QDrant vector store with automatic service installation.
    """
    success = qdrant_url.lower() != 'none' and is_qdrant_running(qdrant_url)
    if not success:
        print(f"Warning: QDrant is not running at {qdrant_url}")
        # Only prompt for localhost (not remote servers)
        if qdrant_url == 'http://localhost:6333':
            print(f"Attempting to install QDrant as a service...")
            if _prompt_qdrant_install():
                if _install_qdrant_service():
                    # Give QDrant a moment to start
                    wait_for_qdrant(qdrant_url, timeout=10)
                    success = True

    if success:
        return MynaVectorStore(url=qdrant_url)
    else:
        raise Exception(f"Could not connect to QDrant at {qdrant_url}")


def _prompt_qdrant_install() -> bool:
    """Ask user if they want to install QDrant as a service"""
    while True:
        response = input("\nWould you like to install QDrant as a startup service? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def _install_qdrant_service() -> bool:
    """Install QDrant service and return success status"""
    import subprocess

    print("\n🚀 Installing QDrant service...")
    try:
        result = subprocess.run(
            ["python", "install_qdrant_service.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("✅ QDrant service installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install QDrant service: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ install_qdrant_service.py not found")
        return False


def get_incomplete_tracks(vector_store: MynaVectorStore) -> List[PointStruct]:
    """
    Get all tracks that need processing
    """
    return vector_store.get_tracks_without_embeddings()


def compute_embeddings(melspecs: torch.Tensor, inference: MynaInference) -> torch.Tensor:
    """
    Compute embeddings for a batch of audio samples.
    """
    with torch.no_grad():
        sample_embeds = []
        for i in range(melspecs.shape[0]):
            sample_ms = melspecs[i].unsqueeze(0).unsqueeze(0)
            embed = inference.model(sample_ms)
            sample_embeds.append(embed)
        return torch.cat(sample_embeds, dim=0)


def process_track(file_path: str, vector_store: MynaVectorStore,
                 inference: MynaInference) -> bool:
    """
    Process a single track, filling in missing data.

    Args:
        file_path: Path to audio file
        vector_store: QDrant vector store
        inference: Myna inference engine

    Returns:
        bool: True if processing succeeded
    """
    try:
        track = vector_store.get_track(file_path)
        if not track:
            print(f"❌ Track not found in database: {file_path}")
            return False

        melspecs, audio_hash, mean_energy, waveform_peaks, duration_seconds = inference.preprocess_audio(file_path)
        embeddings = compute_embeddings(melspecs, inference)

        print(f"Storing track: {file_path}, {track}")
        vector_store.store_track(track,
            audio_hash=audio_hash,
            energy=mean_energy,
            waveform=waveform_peaks,
            duration=duration_seconds,
            embeddings=embeddings,
        )

        return True

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        vector_store.mark_as_failed(track, str(e))
        return False


def compute_pca_for_all(vector_store: MynaVectorStore, debug: bool = False) -> bool:
    """
    Compute PCA vectors for all tracks that have embeddings but no PCA.

    Args:
        vector_store: QDrant vector store

    Returns:
        bool: True if PCA computation succeeded
    """
    points = vector_store.get_tracks_with_embeddings()
    embeddings = np.array([point.vector["embedding768"] for point in points])

    if len(points) == 0:
        print("PCA: Nothing to do, no points to update")
        return False

    # Use min(n_points, 16) components to avoid dimensionality issues
    n_components = min(len(points), 16)
    if debug:
        print(f"PCA: Using {n_components} components")
    pca = PCA(n_components=n_components)
    pca_vectors = pca.fit_transform(embeddings)

    vector_store.update_pca(points, pca_vectors)
    if debug:
        print("✅ PCA vectors and flags updated successfully")
    return True
