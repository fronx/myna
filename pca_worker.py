#!/usr/bin/env python3
"""
MusicMapper PCA Worker

Separate background worker that continuously monitors for tracks with embeddings
but no PCA vectors, and computes PCA for them. Runs independently from the main
embedding worker to provide frequent UI updates.
"""

import argparse
import time
from typing import List
from vector_store import MynaVectorStore
from indexing import connect_to_qdrant, compute_pca_for_all


def check_and_compute_pca(vector_store: MynaVectorStore, debug: bool = False) -> bool:
    if vector_store.pca_required():
        if debug:
            print("PCA computation needed, computing...", flush=True)
        return compute_pca_for_all(vector_store, debug=debug)
    else:
        if debug:
            print("No PCA computation needed", flush=True)
        return False


def run_pca_worker_cycle(vector_store: MynaVectorStore, debug: bool = False) -> None:
    if debug:
        print("PCA worker cycle starting...", flush=True)

    pca_computed = check_and_compute_pca(vector_store, debug=debug)

    if pca_computed and debug:
        print("PCA worker cycle complete - PCA computed", flush=True)
    elif debug:
        print("PCA worker cycle complete - no work needed", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Background PCA worker for computing PCA vectors')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                       help='QDrant server URL (default: http://localhost:6333)')
    parser.add_argument('--daemon', action='store_true',
                       help='Run continuously as daemon')
    parser.add_argument('--sleep-interval', type=int, default=10,
                       help='Sleep interval in daemon mode (seconds, default: 10)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')

    args = parser.parse_args()

    print(f"PCA Worker: Connecting to QDrant at {args.qdrant_url}...", flush=True)
    vector_store = connect_to_qdrant(args.qdrant_url)
    print(f"PCA Worker Connected: {vector_store.collection_info()}", flush=True)

    if args.daemon:
        print(f"PCA Worker: Running in daemon mode (checking every {args.sleep_interval}s)", flush=True)
        print("   Press Ctrl+C to stop", flush=True)

        try:
            while True:
                run_pca_worker_cycle(vector_store, debug=args.debug)
                if args.debug:
                    print(f"PCA Worker: Sleeping for {args.sleep_interval}s...", flush=True)
                time.sleep(args.sleep_interval)

        except KeyboardInterrupt:
            print("\nPCA Worker stopped by user", flush=True)

    else:
        run_pca_worker_cycle(vector_store, debug=args.debug)


if __name__ == '__main__':
    main()