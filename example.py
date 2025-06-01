"""
Minimal script example for model inference
"""

import argparse
import subprocess
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

    args = parser.parse_args()

    inference = MynaInference(
        model_path=args.model_path,
        model_type=args.model_type,
        hybrid_mode=args.hybrid_mode
    )

    # Initialize vector store unless explicitly disabled
    vector_store = None
    if args.qdrant_url.lower() != 'none':
        # Check if QDrant is running
        if not is_qdrant_running(args.qdrant_url):
            # Only prompt for localhost (not remote servers)
            if args.qdrant_url == 'http://localhost:6333':
                if prompt_qdrant_install():
                    if install_qdrant_service():
                        # Give QDrant a moment to start
                        wait_for_qdrant(args.qdrant_url, timeout=10)
                    else:
                        print("Proceeding without vector storage...")
                        args.qdrant_url = 'none'
                else:
                    print("Proceeding without vector storage...")
                    args.qdrant_url = 'none'
            else:
                print(f"Warning: Could not connect to QDrant at {args.qdrant_url}")
                print("Proceeding without vector storage...")
                args.qdrant_url = 'none'
        
        # Try to connect to QDrant
        if args.qdrant_url.lower() != 'none':
            try:
                vector_store = MynaVectorStore(url=args.qdrant_url)
                print(f"Vector store initialized: {vector_store.collection_info()}")
            except Exception as e:
                print(f"Warning: Could not connect to QDrant at {args.qdrant_url}: {e}")
                print("Proceeding without vector storage...")
                vector_store = None

    try:
        audio_files = inference.get_audio_files(args.folder)
        
        # Limit to 25 files for profiling
        if len(audio_files) > 25:
            audio_files = audio_files[:25]
            print(f"Found {len(inference.get_audio_files(args.folder))} audio files, profiling first 25 in {args.folder}")
        else:
            print(f"Found {len(audio_files)} audio files in {args.folder}")

        progress_bar = tqdm(total=len(audio_files), desc="Processing audio files")
        stored_count = 0

        def progress_callback(filename, success, result_or_error, audio_hash=None):
            nonlocal stored_count
            if result_or_error == "skipped":
                tqdm.write(f'⏭ {filename}: already processed (hash match)')
            else:
                embeddings = result_or_error
                tqdm.write(f'✓ {filename}: embeddings shape {embeddings.shape}')
                
                # Store in vector database if enabled
                if vector_store:
                    # Get full file path for the file we just processed
                    full_path = None
                    for audio_file in audio_files:
                        if audio_file.endswith(filename):
                            full_path = audio_file
                            break
                    
                    vector_store.store_track(full_path, embeddings, audio_hash)
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
        
        results = inference.process_folder(args.folder, vector_store, progress_callback, audio_files=audio_files)
        
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
        
        if vector_store:
            print(f"Stored {stored_count} embeddings in vector database")
            print(f"Collection info: {vector_store.collection_info()}")

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()