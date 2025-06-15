import time, sys, torch
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]  # musicmapper.local
sys.path.insert(0, str(repo_root))

from myna.myna_inference import create_inference_engine
engine = create_inference_engine()
audio = sys.argv[1]
t0 = time.perf_counter()
print(f"engine: {engine.preprocess_audio}")
engine.preprocess_audio(audio, profile=True)
print(f"elapsed: {time.perf_counter()-t0:.3f}s")
