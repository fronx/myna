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
import sys
from typing import List, Dict
from myna_inference import create_inference_engine
from indexing import (
    connect_to_qdrant,
    get_incomplete_tracks,
    process_track,
    compute_pca_for_all
)


def process_incomplete_tracks(vector_store, inference, max_tracks: int = None) -> Dict[str, int]:
    """
    Process all incomplete tracks in the database.

    Args:
        vector_store: QDrant vector store
        inference: Myna inference engine
        max_tracks: Maximum number of tracks to process (None = all)

    Returns:
        Dict with processing statistics
    """
    print("🔍 Finding incomplete tracks...")
    incomplete_tracks = get_incomplete_tracks(vector_store)

    if not incomplete_tracks:
        print("✅ All tracks are complete!")
        return {"processed": 0, "failed": 0, "skipped": 0}

    # Limit processing if specified
    if max_tracks and len(incomplete_tracks) > max_tracks:
        incomplete_tracks = incomplete_tracks[:max_tracks]
        print(f"📝 Processing first {max_tracks} of {len(get_incomplete_tracks(vector_store))} incomplete tracks")
    else:
        print(f"📝 Found {len(incomplete_tracks)} incomplete tracks to process")

    stats = {"processed": 0, "failed": 0, "skipped": 0}

    for i, track in enumerate(incomplete_tracks, 1):
        file_path = track["file_path"]
        missing_data = track["missing"]

        if not file_path:
            print(f"❌ Track {i}/{len(incomplete_tracks)}: No file path")
            stats["failed"] += 1
            continue

        print(f"\n📁 Processing {i}/{len(incomplete_tracks)}: {file_path}")
        print(f"   Missing: {', '.join(missing_data)}")

        # Check if file still exists
        import os
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            stats["failed"] += 1
            continue

        # Process the track
        success = process_track(file_path, vector_store, inference, missing_data)

        if success:
            stats["processed"] += 1
            print(f"✅ Completed: {file_path}")
        else:
            stats["failed"] += 1

    return stats


def run_worker_cycle(vector_store, inference, max_tracks: int = None) -> bool:
    """
    Run one complete worker cycle: process incomplete tracks + compute PCA.

    Returns:
        bool: True if any work was done
    """
    print("🚀 Starting worker cycle...")

    # Process incomplete tracks
    stats = process_incomplete_tracks(vector_store, inference, max_tracks)

    work_done = stats["processed"] > 0

    print(f"\n📊 Processing complete:")
    print(f"   ✅ Processed: {stats['processed']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   ⏭ Skipped: {stats['skipped']}")

    # Always try to compute PCA if we have embeddings
    if stats["processed"] > 0:
        print("\n🔬 Computing PCA for newly processed tracks...")
        pca_success = compute_pca_for_all(vector_store)
        if pca_success:
            print("✅ PCA computation complete")
        else:
            print("❌ PCA computation failed")

    remaining = len(get_incomplete_tracks(vector_store))
    if remaining > 0:
        print(f"\n📋 {remaining} tracks still need processing")
        work_done = True  # Still work to do
    else:
        print("\n🎉 All tracks are now complete!")

    return work_done


def main():
    parser = argparse.ArgumentParser(description='Background worker for processing incomplete music tracks')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                       help='QDrant server URL (default: http://localhost:6333)')
    parser.add_argument('--model-path', default='pretrained/myna-hybrid.pth',
                       help='Path to Myna model checkpoint')
    parser.add_argument('--model-type', default='hybrid', choices=['square', 'vertical', 'hybrid'],
                       help='Myna model type')
    parser.add_argument('--daemon', action='store_true',
                       help='Run continuously as daemon, checking for new work every 1 seconds')
    parser.add_argument('--max-tracks', type=int,
                       help='Maximum number of tracks to process per cycle')
    parser.add_argument('--sleep-interval', type=int, default=1,
                       help='Sleep interval in daemon mode (seconds)')

    args = parser.parse_args()

    try:
        # Connect to QDrant
        print(f"🔌 Connecting to QDrant at {args.qdrant_url}...")
        vector_store = connect_to_qdrant(args.qdrant_url)
        print(f"✅ Connected: {vector_store.collection_info()}")

        # Initialize inference engine
        print(f"🧠 Loading Myna model from {args.model_path}...")
        inference = create_inference_engine(
            model_path=args.model_path,
            model_type=args.model_type,
            hybrid_mode=True
        )
        print("✅ Model loaded successfully")

        if args.daemon:
            print(f"🔄 Running in daemon mode (checking every {args.sleep_interval}s)")
            print("   Press Ctrl+C to stop")

            try:
                while True:
                    work_done = run_worker_cycle(vector_store, inference, args.max_tracks)

                    if not work_done:
                        print(f"😴 No work to do, sleeping for {args.sleep_interval}s...")
                    else:
                        print(f"⏱ Cycle complete, sleeping for {args.sleep_interval}s...")

                    time.sleep(args.sleep_interval)

            except KeyboardInterrupt:
                print("\n👋 Worker stopped by user")

        else:
            # Single run mode
            work_done = run_worker_cycle(vector_store, inference, args.max_tracks)

            if work_done:
                remaining = len(get_incomplete_tracks(vector_store))
                if remaining > 0:
                    print(f"\n💡 Run again to process remaining {remaining} tracks")
                    sys.exit(1)  # Exit code indicates more work available

    except Exception as e:
        print(f"❌ Worker error: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()