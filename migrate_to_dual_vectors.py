#!/usr/bin/env python3
"""
Migrate existing myna_embeddings collection to support dual vectors (embedding768 + pca16)
"""

import numpy as np
from sklearn.decomposition import PCA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def migrate_collection():
    client = QdrantClient(url="http://localhost:6333")

    # Get all existing points
    print("Fetching all existing points...")
    all_points = []
    scroll_result = client.scroll(
        collection_name="myna_embeddings",
        limit=100,
        with_payload=True,
        with_vectors=True
    )

    all_points.extend(scroll_result[0])
    next_page = scroll_result[1]

    while next_page is not None:
        scroll_result = client.scroll(
            collection_name="myna_embeddings",
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=True
        )
        all_points.extend(scroll_result[0])
        next_page = scroll_result[1]

    print(f"Retrieved {len(all_points)} points")

    # Extract all embeddings for PCA
    embeddings = np.array([point.vector for point in all_points])
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute PCA
    print("Computing PCA...")
    pca = PCA(n_components=16)
    pca_embeddings = pca.fit_transform(embeddings)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Create new collection with dual vectors
    new_collection = "myna_embeddings_new"
    print(f"Creating new collection: {new_collection}")

    client.create_collection(
        collection_name=new_collection,
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

    # Insert points with both vectors
    print("Inserting points with dual vectors...")
    from qdrant_client.models import PointVectors

    for i, point in enumerate(all_points):
        client.upsert(
            collection_name=new_collection,
            points=[
                {
                    "id": point.id,
                    "vector": {
                        "embedding768": point.vector,
                        "pca16": pca_embeddings[i].tolist()
                    },
                    "payload": point.payload
                }
            ]
        )

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(all_points)} points")

    print("\nMigration complete!")
    print(f"Old collection: myna_embeddings ({len(all_points)} points)")
    print(f"New collection: {new_collection} ({len(all_points)} points with dual vectors)")
    print("\nTo complete migration:")
    print("1. Delete old collection: client.delete_collection('myna_embeddings')")
    print(f"2. Rename new collection: client.update_collection_aliases(change_aliases_operations=[...")
    print("   CreateAliasOperation(create_alias=CreateAlias(collection_name='myna_embeddings_new', alias_name='myna_embeddings'))])")

if __name__ == "__main__":
    migrate_collection()