"""
QDrant vector store for Myna embeddings
"""

import os
import hashlib
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from mutagen import File as MutagenFile


class MynaVectorStore:
    """QDrant vector store for Myna embeddings"""

    def __init__(self, collection_name: str = "myna_embeddings_new",
                 url: Optional[str] = None, db_path: Optional[str] = None):
        """
        Initialize QDrant vector store.

        Args:
            collection_name: Name of the QDrant collection
            url: QDrant server URL (if None, uses local storage)
            db_path: Local database path when url is None (defaults to ~/.qdrant_data)
        """
        self.collection_name = collection_name

        if url:
            self.client = QdrantClient(url=url)
        else:
            if db_path is None:
                from pathlib import Path
                db_path = str(Path.home() / ".qdrant_data")
            self.client = QdrantClient(path=db_path)

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Myna embeddings are 768-dimensional, PCA are 16-dimensional
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "embedding768": VectorParams(
                        size=768,
                        distance=Distance.COSINE
                    ),
                    "pca16": VectorParams(
                        size=16,
                        distance=Distance.COSINE
                    )
                }
            )
            print(f"Created collection: {self.collection_name}")

    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract metadata from audio file"""
        metadata = {"file_path": file_path, "filename": os.path.basename(file_path)}

        try:
            audio_file = MutagenFile(file_path)
            if audio_file is not None:
                # Extract common tags
                metadata.update({
                    "title": self._get_tag(audio_file, "TIT2", "TITLE"),
                    "artist": self._get_tag(audio_file, "TPE1", "ARTIST"),
                    "album": self._get_tag(audio_file, "TALB", "ALBUM"),
                    "genre": self._get_tag(audio_file, "TCON", "GENRE"),
                    "year": self._get_tag(audio_file, "TDRC", "DATE"),
                    "duration": getattr(audio_file.info, 'length', 0),
                    "bitrate": getattr(audio_file.info, 'bitrate', 0),
                })
        except Exception as e:
            print(f"Warning: Could not extract metadata from {file_path}: {e}")

        return metadata

    def _get_tag(self, audio_file, *tag_names):
        """Get tag value from audio file, trying multiple tag formats"""
        for tag_name in tag_names:
            if tag_name in audio_file:
                value = audio_file[tag_name]
                if isinstance(value, list) and value:
                    return str(value[0])
                return str(value)
        return ""

    def _create_file_hash(self, file_path: str) -> str:
        """Create hash of file for deduplication"""
        stat = os.stat(file_path)
        # Hash based on file path, size, and modification time
        hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()


    def _average_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Average multiple embeddings from strategic sampling into single vector.

        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)

        Returns:
            np.ndarray: Averaged embedding vector
        """
        if embeddings.dim() == 2:
            # Multiple samples - average them
            return embeddings.mean(dim=0).cpu().numpy()
        else:
            # Single sample
            return embeddings.cpu().numpy()

    def store_track(self, file_path: str, embeddings: torch.Tensor,
                   audio_hash: str, energy: float, waveform: List, duration: float,
                   pca_embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None) -> str:
        """
        Store track embeddings and metadata.

        Args:
            file_path: Path to audio file
            embeddings: Myna embeddings tensor
            audio_hash: Hash of the audio samples used for embedding generation
            energy: Energy value extracted from audio segments
            waveform: Waveform peaks data for visualization
            duration: Duration of the audio file in seconds
            pca_embedding: Optional PCA-reduced embedding (16D) - computed in second pass
            metadata: Optional additional metadata

        Returns:
            str: Point ID in QDrant
        """
        # Create unique ID based on file
        point_id = self._create_file_hash(file_path)

        # Extract metadata
        track_metadata = self._extract_metadata(file_path)
        if metadata:
            track_metadata.update(metadata)

        # Add audio sample hash, energy, waveform, and duration
        track_metadata["audio_sample_hash"] = audio_hash
        track_metadata["energy"] = energy
        track_metadata["waveform"] = waveform
        track_metadata["duration"] = duration

        # Average embeddings if multiple samples
        embedding_vector = self._average_embeddings(embeddings)

        # Check if point already exists to preserve vectors
        try:
            existing_points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True
            )
            existing_vectors = existing_points[0].vector if existing_points else None
        except:
            existing_vectors = None

        # Prepare vectors - always include embedding768
        vectors = {"embedding768": embedding_vector.tolist()}

        # If updating existing point, preserve pca16 if not provided
        if existing_vectors and isinstance(existing_vectors, dict):
            if pca_embedding is None and "pca16" in existing_vectors:
                vectors["pca16"] = existing_vectors["pca16"]

        # Add PCA if provided (second pass)
        if pca_embedding is not None:
            vectors["pca16"] = pca_embedding.tolist()

        # Store in QDrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=track_metadata
                )
            ]
        )

        return point_id

    def find_similar(self, file_path: str = None, embeddings: torch.Tensor = None,
                    limit: int = 10, score_threshold: float = 0.7,
                    using: str = "embedding768") -> List[Dict]:
        """
        Find similar tracks.

        Args:
            file_path: Path to query audio file (if stored)
            embeddings: Query embeddings tensor (alternative to file_path)
            limit: Number of results to return
            score_threshold: Minimum similarity score
            using: Which vector to use for search ("embedding768" or "pca16")

        Returns:
            List of similar tracks with metadata and scores
        """
        if file_path:
            # Search by stored track
            query_id = self._create_file_hash(file_path)
            query_vector = None
        elif embeddings is not None:
            # Search by embedding vector
            query_id = None
            query_vector = self._average_embeddings(embeddings).tolist()
        else:
            raise ValueError("Either file_path or embeddings must be provided")

        if query_vector:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                using=using,
                limit=limit,
                score_threshold=score_threshold
            )
        else:
            # Get the stored point first
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[query_id],
                with_vectors=[using]
            )

            if not points:
                raise ValueError(f"Track not found: {file_path}")

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=points[0].vector[using],
                using=using,
                limit=limit + 1,  # +1 to exclude self
                score_threshold=score_threshold
            )

            # Remove the query track itself
            results = [r for r in results if r.id != query_id][:limit]

        # Format results
        similar_tracks = []
        for result in results:
            track_info = {
                "id": result.id,
                "score": result.score,
                "metadata": result.payload
            }
            similar_tracks.append(track_info)

        return similar_tracks

    def get_track_info(self, file_path: str) -> Optional[Dict]:
        """Get stored track information"""
        track_id = self._create_file_hash(file_path)
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[track_id],
            with_payload=True
        )

        if points:
            return points[0].payload
        return None

    def delete_track(self, file_path: str) -> bool:
        """Delete track from vector store"""
        track_id = self._create_file_hash(file_path)
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=[track_id]
        )
        return result.operation_id is not None

    def needs_reprocessing(self, file_path: str, current_audio_hash: str) -> bool:
        """
        Check if a file needs reprocessing by comparing audio content hashes and checking for energy data.

        Args:
            file_path: Path to audio file
            current_audio_hash: Current hash of the audio samples

        Returns:
            bool: True if file needs reprocessing, False if already up-to-date
        """
        track_info = self.get_track_info(file_path)

        if not track_info:
            # No existing data, needs processing
            return True

        stored_audio_hash = track_info.get("audio_sample_hash")
        if not stored_audio_hash:
            # No hash stored, needs reprocessing
            return True

        if "energy" not in track_info:
            # Missing energy data, needs reprocessing
            return True

        if "waveform" not in track_info:
            # Missing waveform data, needs reprocessing
            return True

        if "duration" not in track_info:
            # Missing duration data, needs reprocessing
            return True

        # Compare audio content hashes
        return stored_audio_hash != current_audio_hash

    def collection_info(self) -> Dict:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "config": info.config,
            "points_count": info.points_count,
            "status": info.status
        }

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all embedding768 vectors for PCA fitting.

        Returns:
            Tuple of (point_ids, embeddings_array)
        """
        all_points = []
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_vectors=["embedding768"]
        )

        all_points.extend(scroll_result[0])
        next_page = scroll_result[1]

        while next_page is not None:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_page,
                with_vectors=["embedding768"]
            )
            all_points.extend(scroll_result[0])
            next_page = scroll_result[1]

        # Extract IDs and vectors
        point_ids = [point.id for point in all_points]
        embeddings = np.array([point.vector["embedding768"] for point in all_points])

        return point_ids, embeddings

    def update_pca_vectors(self, point_ids: List[str], pca_vectors: np.ndarray):
        """
        Update PCA vectors for multiple points.

        Args:
            point_ids: List of point IDs to update
            pca_vectors: Array of PCA vectors (shape: [n_points, 16])
        """
        # Update in batches - use update_vectors to preserve existing vectors
        batch_size = 100
        for i in range(0, len(point_ids), batch_size):
            batch_ids = point_ids[i:i + batch_size]
            batch_vectors = pca_vectors[i:i + batch_size]

            # Create update points with only the pca16 vector
            points = []
            for j, point_id in enumerate(batch_ids):
                points.append({
                    "id": point_id,
                    "vector": {"pca16": batch_vectors[j].tolist()}
                })

            # Use update_vectors to only update pca16, preserving embedding768
            self.client.update_vectors(
                collection_name=self.collection_name,
                points=points
            )