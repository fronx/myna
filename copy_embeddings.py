#!/usr/bin/env python3
"""
Copy embedding768 vectors from myna_embeddings to myna_embeddings_new collection.
Verifies file_path matches before copying.
"""

import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from tqdm import tqdm

def copy_embeddings():
    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    # Get collection info
    old_collection = "myna_embeddings"
    new_collection = "myna_embeddings_new"
    
    try:
        old_info = client.get_collection(old_collection)
        new_info = client.get_collection(new_collection)
        print(f"Old collection '{old_collection}' has {old_info.points_count} points")
        print(f"New collection '{new_collection}' has {new_info.points_count} points")
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return
    
    # Scroll through all points in both collections
    print("\nFetching all points from both collections...")
    
    # Get all points from old collection
    old_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=old_collection,
            scroll_filter=None,
            limit=100,
            with_payload=True,
            with_vectors=True,  # Get all vectors (unnamed in old collection)
            offset=offset
        )
        old_points.extend(result[0])
        offset = result[1]
        if offset is None:
            break
    
    print(f"Retrieved {len(old_points)} points from old collection")
    
    # Get all points from new collection (without vectors, just payloads)
    new_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=new_collection,
            scroll_filter=None,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )
        new_points.extend(result[0])
        offset = result[1]
        if offset is None:
            break
    
    print(f"Retrieved {len(new_points)} points from new collection")
    
    # Create mapping of file_path to point for new collection
    new_points_by_path = {}
    for point in new_points:
        if point.payload and "file_path" in point.payload:
            file_path = point.payload["file_path"]
            if file_path in new_points_by_path:
                print(f"Warning: Duplicate file_path found: {file_path}")
            new_points_by_path[file_path] = point
    
    print(f"\nFound {len(new_points_by_path)} unique file paths in new collection")
    
    # Copy embedding768 vectors
    print("\nCopying embedding768 vectors...")
    copied_count = 0
    missing_count = 0
    error_count = 0
    
    for old_point in tqdm(old_points, desc="Copying embeddings"):
        if not old_point.payload or "file_path" not in old_point.payload:
            print(f"Warning: Old point {old_point.id} has no file_path")
            continue
        
        file_path = old_point.payload["file_path"]
        
        # Find corresponding point in new collection
        if file_path not in new_points_by_path:
            missing_count += 1
            print(f"Warning: File path not found in new collection: {file_path}")
            continue
        
        new_point = new_points_by_path[file_path]
        
        # Check if old point has vector (unnamed in old collection)
        if not old_point.vector:
            print(f"Warning: Old point for {file_path} has no vector")
            continue
        
        # Get the vector - it's unnamed in old collection, so it's directly the vector
        vector_data = old_point.vector
        if isinstance(vector_data, dict):
            # If it's a dict, it might have named vectors
            if "embedding768" in vector_data:
                embedding_vector = vector_data["embedding768"]
            else:
                print(f"Warning: Old point for {file_path} has named vectors but no embedding768")
                continue
        else:
            # It's an unnamed vector, use it directly
            embedding_vector = vector_data
        
        # Update the new point with the embedding768 vector
        try:
            client.update_vectors(
                collection_name=new_collection,
                points=[{
                    "id": new_point.id,
                    "vector": {
                        "embedding768": embedding_vector
                    }
                }]
            )
            copied_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error updating point {new_point.id} for {file_path}: {e}")
    
    print(f"\n✅ Successfully copied {copied_count} embedding768 vectors")
    if missing_count > 0:
        print(f"⚠️  {missing_count} file paths from old collection not found in new collection")
    if error_count > 0:
        print(f"❌ {error_count} errors occurred during copying")
    
    # Verify a few random points
    print("\nVerifying a few random points...")
    import random
    sample_paths = random.sample(list(new_points_by_path.keys()), min(5, len(new_points_by_path)))
    
    for path in sample_paths:
        new_point_id = new_points_by_path[path].id
        result = client.retrieve(
            collection_name=new_collection,
            ids=[new_point_id],
            with_vectors=True
        )
        if result and result[0].vector:
            vectors = list(result[0].vector.keys())
            print(f"✓ {path}: has vectors {vectors}")
        else:
            print(f"✗ {path}: missing vectors")

if __name__ == "__main__":
    copy_embeddings()