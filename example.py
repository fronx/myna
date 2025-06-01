"""
Minimal script example for model inference
"""

import argparse
from tqdm import tqdm
from myna_inference import MynaInference


def main():
    parser = argparse.ArgumentParser(description='Process audio files in a folder with Myna model')
    parser.add_argument('folder', help='Path to folder containing audio files')
    parser.add_argument('--model-path', default='pretrained/myna-hybrid.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', default='hybrid', choices=['square', 'vertical', 'hybrid'],
                       help='Model type')
    parser.add_argument('--hybrid-mode', action='store_true', default=True,
                       help='Concatenate embeddings for hybrid models; disable to only use square patches')

    args = parser.parse_args()

    inference = MynaInference(
        model_path=args.model_path,
        model_type=args.model_type,
        hybrid_mode=args.hybrid_mode
    )

    try:
        audio_files = inference.get_audio_files(args.folder)
        print(f"Found {len(audio_files)} audio files in {args.folder}")

        progress_bar = tqdm(total=len(audio_files), desc="Processing audio files")

        def progress_callback(filename, success, result_or_error):
            if success:
                tqdm.write(f'✓ {filename}: embeddings shape {result_or_error.shape}')
            else:
                tqdm.write(f'✗ {filename}: {result_or_error}')
            progress_bar.update(1)

        results = inference.process_folder(args.folder, progress_callback)
        progress_bar.close()

        print(f"\nProcessed {len([r for r in results.values() if r is not None])} files successfully")

    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()