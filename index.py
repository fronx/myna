"""
CLI script for processing audio files and generating embeddings
"""

import argparse
import cProfile
import pstats
import io
from tqdm import tqdm
from myna_inference import MynaInference
from indexing import connect_to_qdrant, compute_pca_for_all

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

        if vector_store and successful_files > 0:
            if stored_count > 0:
                print(f"Stored {stored_count} new embeddings in vector database")

            # Compute PCA using the library function
            compute_pca_for_all(vector_store)
            print(f"\nCollection info: {vector_store.collection_info()}")

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()