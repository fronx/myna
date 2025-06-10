#!/usr/bin/env python3
"""
MusicMapper Background Worker

Processes incomplete tracks in QDrant, filling in missing:
- Metadata extraction
- Waveform generation
- Embedding computation
- PCA vector computation
"""

import argparse
import time
import os
from typing import Dict
from myna_inference import create_inference_engine, MynaInference
from vector_store import MynaVectorStore
from indexing import (
    connect_to_qdrant,
    get_incomplete_tracks,
    process_track,
    compute_pca_for_all
)


def process_incomplete_tracks(vector_store: MynaVectorStore, inference: MynaInference) -> Dict[str, int]:
    """
    Process all incomplete tracks in the database.

    Args:
        vector_store: QDrant vector store
        inference: Myna inference engine

    Returns:
        Dict with processing statistics
    """
    tracks = get_incomplete_tracks(vector_store)
    stats = {"processed": 0, "failed": 0}

    if not tracks:
        return stats

    print(f"Found {len(tracks)} incomplete tracks to process", flush=True)
    for i, track in enumerate(tracks, 1):
        file_path = track.payload["file_path"]

        print(f"\nProcessing {i}/{len(tracks)}: {file_path}", flush=True)

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}", flush=True)
            stats["failed"] += 1
            continue

        if track.payload.get("error"):
            print(f"Skipping track that was marked as failed previously: {file_path}", flush=True)
            stats["failed"] += 1
            continue

        success = process_track(file_path, vector_store, inference)

        if success:
            stats["processed"] += 1
            print(f"✅ Completed: {file_path}", flush=True)
        else:
            stats["failed"] += 1

    return stats


def run_worker_cycle(vector_store, inference):
    """
    Run one complete worker cycle: process incomplete tracks + compute PCA.
    """
    # print("Starting worker cycle...", flush=True)
    stats = process_incomplete_tracks(vector_store, inference)

    if stats["failed"] > 0 or stats["processed"] > 0:
        print(f"\nProcessing complete:", flush=True)
        print(f"   ✅ Processed: {stats['processed']}", flush=True)
        print(f"   ❌ Failed: {stats['failed']}", flush=True)

    if stats["processed"] > 0:
        compute_pca_for_all(vector_store, debug=True)


def main():
    parser = argparse.ArgumentParser(description='Background worker for processing incomplete music tracks')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                       help='QDrant server URL (default: http://localhost:6333)')
    parser.add_argument('--model-path', default='pretrained/myna-hybrid.pth',
                       help='Path to Myna model checkpoint')
    parser.add_argument('--model-type', default='hybrid', choices=['square', 'vertical', 'hybrid'],
                       help='Myna model type')
    parser.add_argument('--daemon', action='store_true',
                       help='Run continuously as daemon')
    parser.add_argument('--sleep-interval', type=int, default=1,
                       help='Sleep interval in daemon mode (seconds)')

    args = parser.parse_args()

    print(f"Connecting to QDrant at {args.qdrant_url}...", flush=True)
    vector_store = connect_to_qdrant(args.qdrant_url)
    print(f"Connected: {vector_store.collection_info()}", flush=True)

    print(f"Loading Myna model from {args.model_path}...", flush=True)
    inference = create_inference_engine(
        model_path=args.model_path,
        model_type=args.model_type,
        hybrid_mode=True
    )
    print("Model loaded successfully", flush=True)

    if args.daemon:
        print(f"Running in daemon mode", flush=True)
        print("   Press Ctrl+C to stop", flush=True)

        try:
            while True:
                run_worker_cycle(vector_store, inference)
                print(f"...", flush=True)
                time.sleep(args.sleep_interval)

        except KeyboardInterrupt:
            print("\nWorker stopped by user", flush=True)

    else:
        run_worker_cycle(vector_store, inference)


if __name__ == '__main__':
    main()