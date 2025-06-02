# Myna Automatic Music Embedder

This system automatically processes music collections to extract semantic embeddings, audio features, and waveform data for music analysis and recommendation systems.

## What It Does

The Myna embedder takes a folder of audio files and automatically:

1. **Extracts Semantic Embeddings**: Uses the Myna transformer model to create 768-dimensional semantic embeddings that capture musical similarity
2. **Computes Audio Features**: Extracts energy levels and other audio characteristics 
3. **Generates Waveform Data**: Creates peaks data for frontend visualization (WaveSurfer.js compatible)
4. **Stores Everything**: Saves all data to QDrant vector database for similarity search and retrieval
5. **Handles Deduplication**: Uses content-based hashing to avoid reprocessing unchanged files

## Core Components

### MynaInference (`myna_inference.py`)
- **Strategic Sampling**: Extracts 4 audio segments from strategic positions (15%, 35%, 55%, 75%) to capture representative content
- **Mel Spectrogram Processing**: Converts audio to mel spectrograms optimized for the Myna model
- **Content Hashing**: Creates SHA-256 hashes of audio content for change detection
- **Batch Processing**: Efficiently processes multiple files with progress tracking

### Audio Processing (`audio_utils.py`)
- **Multi-format Support**: Handles MP3, WAV, FLAC, M4A, AAC, OGG files
- **Smart Audio Loading**: Optimized loading with fallbacks (torchaudio → librosa)
- **Waveform Peak Computation**: Vectorized peak extraction for visualization
- **Energy Extraction**: Uses Essentia for perceptual energy analysis

### Vector Storage (`vector_store.py`)
- **QDrant Integration**: Stores embeddings in high-performance vector database
- **Rich Metadata**: Extracts and stores artist, album, genre, duration, etc.
- **Similarity Search**: Enables finding similar tracks by embeddings or file path
- **Smart Reprocessing**: Only reprocesses files when audio content changes

## Processing Pipeline

```
Audio File → Strategic Sampling → Mel Spectrogram → Myna Model → 768D Embedding
    ↓              ↓                    ↓                ↓
Metadata ← Audio Features ← Waveform Peaks ← Content Hash
    ↓
QDrant Vector Database
```

### 1. Audio Analysis
- Loads audio at 16kHz sample rate
- Extracts 4 strategic segments to capture musical diversity
- Computes mel spectrograms (128 mel bins, 96 frames per segment)
- Generates content hash for change detection

### 2. Feature Extraction
- **Embeddings**: 768-dimensional vectors from Myna transformer model
- **Energy**: Normalized energy levels using Essentia
- **Waveform**: Peak data for visualization (configurable samples per pixel)
- **Duration**: Audio length in seconds
- **Metadata**: Artist, album, genre, year, bitrate from file tags

### 3. Storage & Indexing
- Stores embeddings as 768D vectors with cosine similarity
- Saves all metadata and features as payload data
- Enables fast similarity search and retrieval
- Supports incremental updates (only changed files reprocessed)

## Model Architecture

The system supports three Myna model variants:
- **Square Patches**: 16x16 patch size for standard processing
- **Vertical Patches**: 128x2 patch size optimized for frequency analysis  
- **Hybrid Mode**: Combines both patch types for enhanced representation

Default configuration uses hybrid mode with concatenated embeddings for maximum musical understanding.

## Key Features

### Intelligent Reprocessing
- Content-based hashing detects actual audio changes
- Skips unchanged files even if metadata differs
- Forces reprocessing when missing required fields (energy, waveform, duration)

### Optimized Performance
- Vectorized operations for waveform computation
- Strategic sampling reduces processing time vs. full-file analysis
- Efficient memory usage with segment-based processing
- Progress tracking and profiling for bottleneck identification

### Production Ready
- Handles corrupted files gracefully
- Comprehensive error handling and logging
- Metadata extraction with format fallbacks
- QDrant service auto-installation and health checking

## Use Cases

1. **Music Recommendation**: Find similar tracks using semantic embeddings
2. **Music Analysis**: Analyze collections by energy, genre, features
3. **Playlist Generation**: Create playlists based on musical similarity
4. **Music Visualization**: Frontend waveform display with peaks data
5. **Content Organization**: Automatic clustering and categorization
6. **Duplicate Detection**: Identify similar or duplicate tracks

## Output Data Structure

Each processed track contains:
```json
{
  "embedding": [768 float values],
  "metadata": {
    "file_path": "/path/to/track.mp3",
    "title": "Song Title",
    "artist": "Artist Name", 
    "album": "Album Name",
    "genre": "Genre",
    "year": "2023",
    "duration": 245.6,
    "bitrate": 320,
    "energy": 0.0342,
    "waveform": [[peak1, peak2, ...]], 
    "audio_sample_hash": "sha256_hash"
  }
}
```

## Performance

The system is optimized for large music collections:
- **Strategic sampling**: 4 segments vs. full-file processing
- **Vectorized operations**: NumPy-optimized peak computation
- **Content hashing**: Avoids unnecessary reprocessing
- **Batch processing**: Efficient memory and GPU utilization

Typical processing speed: 2-5 seconds per track on modern hardware, with most time spent on model inference rather than audio processing.