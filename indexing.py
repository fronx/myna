"""
MusicMapper Indexing Library

Reusable functions for audio processing, embedding generation,
and QDrant operations. Extracted from index.py to enable
background worker and other scripts.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from typing import Optional, Dict, List, Tuple, Callable
import torch

from myna_inference import MynaInference, create_inference_engine
from vector_store import MynaVectorStore
from qdrant_utils import is_qdrant_running, wait_for_qdrant


def connect_to_qdrant(qdrant_url: str) -> MynaVectorStore:
    """
    Connect to QDrant vector store with automatic service installation.

    Args:
        qdrant_url: QDrant server URL

    Returns:
        MynaVectorStore: Connected vector store instance

    Raises:
        Exception: If connection fails
    """
    success = qdrant_url.lower() != 'none' and is_qdrant_running(qdrant_url)
    if not success:
        print(f"XX QDrant is not running at {qdrant_url}")
        # Only prompt for localhost (not remote servers)
        if qdrant_url == 'http://localhost:6333':
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
    print("\n🔍 QDrant vector database is not running.")
    print("QDrant enables:")
    print("  • Storing music embeddings permanently")
    print("  • Finding similar tracks across your collection")
    print("  • Building music recommendation systems")
    print("  • Data visualization and analysis")

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
        print("✅ QDrant service installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install QDrant service: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ install_qdrant_service.py not found")
        return False


def add_files_to_queue(folder_path: str, vector_store: MynaVectorStore) -> List[str]:
    """
    Add audio files to QDrant as incomplete records (filename/path only).

    Args:
        folder_path: Path to folder containing audio files
        vector_store: QDrant vector store

    Returns:
        List[str]: List of file paths that were added
    """
    # Use existing audio file discovery from myna_inference
    inference = create_inference_engine()  # Just for file discovery
    audio_files = inference.get_audio_files(folder_path)

    added_files = []
    for file_path in audio_files:
        # Check if file already exists in QDrant
        existing = vector_store.get_track_info(file_path)
        if not existing:
            # Add minimal record with just file info
            metadata = {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "has_embeddings": False,  # For efficient filtering
                "has_pca": False          # For efficient filtering
            }

            # Create a dummy point with no vectors (will be filled by worker)
            from qdrant_client.models import PointStruct
            point_id = vector_store._create_file_hash(file_path)

            vector_store.client.upsert(
                collection_name=vector_store.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={},  # No vectors yet
                        payload=metadata
                    )
                ]
            )
            added_files.append(file_path)

    return added_files


def get_incomplete_tracks(vector_store: MynaVectorStore) -> List[Dict]:
    """
    Get all tracks that need processing (missing any required data).
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    incomplete_tracks = []

    # Find tracks that need embeddings (has_embeddings = false)
    filter_no_embeddings = Filter(
        must=[
            FieldCondition(key="has_embeddings", match=MatchValue(value=False))
        ]
    )

    result = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        scroll_filter=filter_no_embeddings,
        limit=1000,
        with_payload=True
    )

    for point in result[0]:
        missing_data = []
        payload = point.payload or {}

        # Determine what's missing based on payload content
        if not payload.get("title"):  # Missing metadata
            missing_data.append("metadata")
        if not payload.get("waveform"):  # Missing waveform
            missing_data.append("waveform")
        missing_data.append("embedding768")  # We know this is missing from filter
        missing_data.append("pca16")  # No embeddings means no PCA either

        incomplete_tracks.append({
            "id": point.id,
            "file_path": payload.get("file_path"),
            "missing": missing_data,
            "payload": payload
        })

    # Find tracks that have embeddings but need PCA (has_embeddings = true, has_pca = false)
    filter_needs_pca = Filter(
        must=[
            FieldCondition(key="has_embeddings", match=MatchValue(value=True)),
            FieldCondition(key="has_pca", match=MatchValue(value=False))
        ]
    )

    result = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        scroll_filter=filter_needs_pca,
        limit=1000,
        with_payload=True
    )

    for point in result[0]:
        incomplete_tracks.append({
            "id": point.id,
            "file_path": point.payload.get("file_path"),
            "missing": ["pca16"],  # Only missing PCA
            "payload": point.payload
        })

    return incomplete_tracks


def process_track(file_path: str, vector_store: MynaVectorStore,
                 inference: MynaInference, missing_data: List[str]) -> bool:
    """
    Process a single track, filling in missing data.

    Args:
        file_path: Path to audio file
        vector_store: QDrant vector store
        inference: Myna inference engine
        missing_data: List of data types to process

    Returns:
        bool: True if processing succeeded
    """
    try:
        # Get current track info
        current_info = vector_store.get_track_info(file_path)
        if not current_info:
            print(f"❌ Track not found in database: {file_path}")
            return False

        # Only process if we need embeddings or waveform (metadata is handled separately)
        if "embedding768" in missing_data or "waveform" in missing_data:
            # Process audio file to get embeddings and waveform
            ms, audio_hash, mean_energy, waveform_peaks, duration_seconds = inference._preprocess_audio(file_path)

            # Run inference to get embeddings
            with torch.no_grad():
                sample_embeds = []
                for i in range(ms.shape[0]):
                    sample_ms = ms[i].unsqueeze(0).unsqueeze(0)
                    embed = inference.model(sample_ms)
                    sample_embeds.append(embed)
                embeddings = torch.cat(sample_embeds, dim=0)

            # Store the track with all data
            vector_store.store_track(
                file_path=file_path,
                embeddings=embeddings,
                audio_hash=audio_hash,
                energy=mean_energy,
                waveform=waveform_peaks,
                duration=duration_seconds
            )

        # Update flags to indicate embeddings are now available
        vector_store.client.set_payload(
            collection_name=vector_store.collection_name,
            payload={"has_embeddings": True},
            points=[vector_store._create_file_hash(file_path)]
        )

        return True

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        # Mark as failed (keep has_embeddings = False)
        try:
            vector_store.client.set_payload(
                collection_name=vector_store.collection_name,
                payload={"error": str(e)},
                points=[vector_store._create_file_hash(file_path)]
            )
        except:
            pass
        return False


def compute_pca_for_all(vector_store: MynaVectorStore) -> bool:
    """
    Compute PCA vectors for all tracks that have embeddings but no PCA.

    Args:
        vector_store: QDrant vector store

    Returns:
        bool: True if PCA computation succeeded
    """
    try:
        print("🔬 Computing PCA vectors...")

        # Get all embeddings
        point_ids, embeddings = vector_store.get_all_embeddings()
        print(f"Retrieved {len(point_ids)} embeddings for PCA")

        if len(point_ids) == 0:
            print("No embeddings found in database")
            return False

        # Fit PCA
        pca = PCA(n_components=16)
        pca_vectors = pca.fit_transform(embeddings)

        # Show explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance ratio: {explained_var:.3f}")
        print(f"PCA shape: {pca_vectors.shape}")

        # Update all tracks with PCA vectors
        print("Updating tracks with PCA vectors...")
        vector_store.update_pca_vectors(point_ids, pca_vectors)
        
        # Set has_pca flag for all updated tracks
        for point_id in point_ids:
            vector_store.client.set_payload(
                collection_name=vector_store.collection_name,
                payload={"has_pca": True},
                points=[point_id]
            )
        print("✅ PCA vectors and flags updated successfully")

        return True

    except Exception as e:
        print(f"❌ Error computing PCA: {e}")
        return False


def get_track_status(track_info: Dict) -> str:
    """
    Determine track processing status based on available data.

    Args:
        track_info: Track information from QDrant

    Returns:
        str: Status string ("ready", "pca_pending", "processing", "queued", "failed")
    """
    if not track_info:
        return "not_found"

    # Check for processing errors
    if "error" in track_info:
        return "failed"

    # Use boolean flags for efficiency, fall back to vector checking
    has_pca = track_info.get("has_pca", False) or ("pca16" in track_info.get("vectors", {}))
    has_embeddings = track_info.get("has_embeddings", False) or ("embedding768" in track_info.get("vectors", {}))
    has_waveform = "waveform" in track_info
    has_metadata = bool(track_info.get("title"))

    if has_pca:
        return "ready"
    elif has_embeddings:
        return "pca_pending"  # Has embeddings but no PCA yet
    elif has_waveform or has_metadata:
        return "processing"  # Has some data, still computing embeddings
    else:
        return "queued"


# Progress callback type for compatibility
ProgressCallback = Callable[[str, bool, any, str, float, List, float], None]


def create_simple_progress_callback(verbose: bool = True) -> ProgressCallback:
    """
    Create a simple progress callback for use with processing functions.

    Args:
        verbose: Whether to print progress messages

    Returns:
        ProgressCallback: Callback function
    """
    def progress_callback(filename: str, success: bool, result_or_error: any,
                         audio_hash: str, mean_energy: float, waveform_peaks: List,
                         duration_seconds: float):
        if not verbose:
            return

        if result_or_error == "skipped":
            print(f'⏭ {filename}: already processed (hash match)')
        else:
            embeddings = result_or_error
            energy_info = f" (energy: {mean_energy:.3f})" if mean_energy is not None else ""
            waveform_info = f", waveform: {len(waveform_peaks[0]) if waveform_peaks else 0} peaks" if waveform_peaks else ""
            duration_info = f", duration: {duration_seconds:.1f}s" if duration_seconds is not None else ""
            print(f'✓ {filename}: embeddings shape {embeddings.shape}{energy_info}{waveform_info}{duration_info}')

    return progress_callback