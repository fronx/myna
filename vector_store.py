"""
QDrant vector store for Myna embeddings
"""

import os
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.conversions.common_types import PointId, Points
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
        # Hash based on file path, size, and modification time (integer seconds)
        mtime_seconds = int(stat.st_mtime)
        hash_input = f"{file_path}_{stat.st_size}_{mtime_seconds}"
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

    def exists(self, file_path: str) -> bool:
        """Check if track exists in vector store"""
        point_id = self._create_file_hash(file_path)
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=False
        )
        return len(points) > 0


    def store_track(self, track: PointStruct, audio_hash: str, energy: float, waveform: List, duration: float,
                   embeddings: torch.Tensor) -> str:
        track.payload.update(self._extract_metadata(track.payload["file_path"]))
        track.payload["audio_sample_hash"] = audio_hash
        track.payload["energy"] = energy
        track.payload["waveform"] = waveform
        track.payload["duration"] = duration

        # Average embeddings if multiple samples
        embedding_vector = self._average_embeddings(embeddings)
        track.vector["embedding768"] = embedding_vector.tolist()
        track.payload["has_embeddings"] = True

        self.client.upsert(
            collection_name=self.collection_name,
            points=[track]
        )

    def mark_as_failed(self, track: PointStruct, error: str) -> str:
        track.payload["error"] = error
        self.client.upsert(
            collection_name=self.collection_name,
            points=[track]
        )

    def update_pca(self, tracks: List[PointStruct], pca_vectors: np.ndarray) -> None:
        """Update PCA vectors for a list of tracks"""
        for i, track in enumerate(tracks):
            track.vector["pca16"] = pca_vectors[i].tolist()
            track.payload["has_pca"] = True

        # Process tracks in batches of 100 to avoid timeouts
        batch_size = 5
        for i in range(0, len(tracks), batch_size):
            batch = tracks[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )


    def _record_to_point_struct(self, record) -> PointStruct:
        """Convert a Record to a PointStruct"""
        return PointStruct(
            id=record.id,
            vector=record.vector or {},
            payload=record.payload or {}
        )

    def get_track(self, file_path: str) -> Optional[PointStruct]:
        """Get point by file path"""
        point_id = self._create_file_hash(file_path)
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True
        )
        if records:
            return self._record_to_point_struct(records[0])
        return None


    def delete_track(self, file_path: str) -> bool:
        """Delete track from vector store"""
        track_id = self._create_file_hash(file_path)
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=[track_id]
        )
        return result.operation_id is not None


    def collection_info(self) -> Dict:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "config": info.config,
            "points_count": info.points_count,
            "status": info.status
        }

    def pca_required(self) -> bool:
        """Check if PCA is required for the collection"""
        count_result = self.client.count(self.collection_name, Filter(
            must=[
                FieldCondition(key="has_pca", match=MatchValue(value=False))
            ]
        ))
        return count_result.count > 0

    def get_tracks_without_embeddings(self) -> List[PointStruct]:
        """
        Get all tracks without embeddings.
        """
        return self.get_filtered_tracks("has_embeddings", MatchValue(value=False))


    def get_tracks_with_embeddings(self) -> List[PointStruct]:
        """
        Get all tracks with embeddings.
        """
        return self.get_filtered_tracks("has_embeddings", MatchValue(value=True))


    def get_filtered_tracks(self, key: str, match: MatchValue) -> List[PointStruct]:
        """
        Get all tracks that match the filter.
        """
        def scroll_tracks(offset: Optional[PointId]) -> Tuple[List[PointStruct], Optional[PointId]]:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key=key, match=match),
                    ]
                ),
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=True,
            )
            # Convert Records to PointStructs
            point_structs = [self._record_to_point_struct(record) for record in records]
            return point_structs, next_offset

        tracks = []
        scroll_result = scroll_tracks(None)
        tracks.extend(scroll_result[0])
        next_page = scroll_result[1]
        while next_page is not None:
            scroll_result = scroll_tracks(next_page)
            tracks.extend(scroll_result[0])
            next_page = scroll_result[1]

        return tracks