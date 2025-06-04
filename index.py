"""
Minimal script example for model inference
"""

import argparse
import subprocess
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from myna_inference import MynaInference
from vector_store import MynaVectorStore
from qdrant_utils import is_qdrant_running, wait_for_qdrant


def prompt_qdrant_install() -> bool:
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


def install_qdrant_service() -> bool:
    """Install QDrant service and return success status"""
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

def connect_to_qdrant(qdrant_url: str) -> MynaVectorStore:
    success = qdrant_url.lower() != 'none' and is_qdrant_running(qdrant_url)
    if not success:
        print(f"XX QDrant is not running at {qdrant_url}")
        # Only prompt for localhost (not remote servers)
        if qdrant_url == 'http://localhost:6333':
            if prompt_qdrant_install():
                if install_qdrant_service():
                    # Give QDrant a moment to start
                    wait_for_qdrant(qdrant_url, timeout=10)
                    success = True

    if success:
        return MynaVectorStore(url=qdrant_url)
    else:
        raise Exception(f"Could not connect to QDrant at {qdrant_url}")

def main():
    parser = argparse.ArgumentParser(description='Process audio files in a folder with Myna model')
    parser.add_argument('folder', help='Path to folder containing audio files')
    parser.add_argument('--model-path', default='pretrained/myna-hybrid.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', default='hybrid', choices=['square', 'vertical', 'hybrid'],
                       help='Model type')
    parser.add_argument('--hybrid-mode', action='store_true', default=True,
                       help='Concatenate embeddings for hybrid models; disable to only use square patches')
    parser.add_argument('--qdrant-url', default='http://localhost:6333',
                       help='QDrant server URL (default: http://localhost:6333, set to "none" to disable)')
    parser.add_argument('--force', action='store_true',
                       help='Force recomputation of all tracks, ignoring existing data')

    args = parser.parse_args()

    inference = MynaInference(
        model_path=args.model_path,
        model_type=args.model_type,
        hybrid_mode=args.hybrid_mode
    )

    # Initialize vector store unless explicitly disabled
    vector_store = connect_to_qdrant(args.qdrant_url)
    print(f"Vector store initialized: {vector_store.collection_info()}")

    try:
        audio_files = inference.get_audio_files(args.folder)
        print(f"Found {len(audio_files)} audio files in {args.folder}")

        progress_bar = tqdm(total=len(audio_files), desc="Processing audio files")
        stored_count = 0

        def progress_callback(filename, success, result_or_error, audio_hash, mean_energy, waveform_peaks=None, duration_seconds=None):
            nonlocal stored_count
            if result_or_error == "skipped":
                tqdm.write(f'⏭ {filename}: already processed (hash match)')
            else:
                embeddings = result_or_error
                energy_info = f" (energy: {mean_energy:.3f})" if mean_energy is not None else ""
                waveform_info = f", waveform: {len(waveform_peaks[0]) if waveform_peaks else 0} peaks" if waveform_peaks else ""
                duration_info = f", duration: {duration_seconds:.1f}s" if duration_seconds is not None else ""
                tqdm.write(f'✓ {filename}: embeddings shape {embeddings.shape}{energy_info}{waveform_info}{duration_info}')

                # Store in vector database if enabled
                if vector_store:
                    # Get full file path for the file we just processed
                    full_path = None
                    for audio_file in audio_files:
                        if audio_file.endswith(filename):
                            full_path = audio_file
                            break

                    vector_store.store_track(full_path, embeddings, audio_hash, mean_energy, waveform_peaks, duration_seconds)
                    stored_count += 1
                    tqdm.write(f'  → Stored in vector database')
            progress_bar.update(1)

        # Add profiling to identify bottlenecks in reprocessing check
        import cProfile
        import pstats
        import io

        print("\n🔍 Profiling to identify preprocessing bottlenecks...")
        profiler = cProfile.Profile()
        profiler.enable()

        results = inference.process_folder(args.folder, vector_store, progress_callback, audio_files=audio_files, force=args.force)

        profiler.disable()
        progress_bar.close()

        # Show top time-consuming functions
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions

        print("\n📊 TOP 10 FUNCTIONS BY TIME:")
        print("="*50)
        print(s.getvalue())

        successful_files = len([r for r in results.values() if r is not None])
        print(f"\nProcessed {successful_files} files successfully")

        if vector_store and stored_count > 0:
            print(f"Stored {stored_count} embeddings in vector database")

            # Compute and update PCA vectors
            print("\n🔬 Computing PCA vectors...")
            try:
                # Get all embeddings
                point_ids, embeddings = vector_store.get_all_embeddings()
                print(f"Retrieved {len(point_ids)} embeddings for PCA")

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
                print("✅ PCA vectors updated successfully")

            except Exception as e:
                print(f"❌ Error computing PCA vectors: {e}")

            print(f"\nCollection info: {vector_store.collection_info()}")

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()