# YouTube Video Transcription Pipeline

A comprehensive pipeline for downloading YouTube videos, segmenting the audio into chunks, and generating transcriptions using state-of-the-art speech recognition.

## Features

- Downloads audio from YouTube videos
- Segments audio using Voice Activity Detection (VAD)
- Transcribes audio segments using NVIDIA's Canary-1B ASR model
- Generates organized output with timestamps and metadata
- Supports batch processing for efficient transcription
- Comprehensive logging and error handling

## Requirements

### Python Version
- Python 3.8 or higher
- CUDA Toolkit 11.8 or compatible version (for GPU support)

### Dependencies
```bash
# Core dependencies
torch>=1.9.0
yt-dlp
pydub
silero-vad
soundfile
ffmpeg-python

# System requirements
ffmpeg
```

## Installation

1. Clone the repository:
```bash
git clone [[repository-url](https://github.com/Koosh0610/adalat-ai-task/)]
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install NeMo Toolkit:
```bash
# Install Cython first
pip install Cython

# Install nemo_toolkit with all dependencies
pip install nvidia-pyindex
pip install nemo_toolkit[all]

# If you encounter any issues, try the following alternative installation:
pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
```

5. Install remaining dependencies:
```bash
pip install yt-dlp pydub silero-vad soundfile ffmpeg-python
```

6. Install FFmpeg:
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Windows**: 
  - Download from [FFmpeg website](https://ffmpeg.org/download.html)
  - Add FFmpeg to your system PATH

### Troubleshooting NeMo Installation

If you encounter issues installing NeMo, try these steps:

1. Ensure CUDA toolkit is properly installed:
```bash
# For Ubuntu
sudo apt-get install nvidia-cuda-toolkit
```

2. Check CUDA version compatibility:
```bash
nvidia-smi
nvcc --version
```

3. Clear pip cache if needed:
```bash
pip cache purge
```

4. Install specific versions if needed:
```bash
# Install specific torch version
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Then install NeMo
pip install nemo_toolkit[all]==1.20.0
```

5. Common Issues:
   - If you see "CUDA out of memory" errors, reduce batch size in the code
   - If you see "Cannot import name 'MelAudioPreprocessor'" error, try reinstalling NeMo
   - For "ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS'" error:
     ```bash
     pip uninstall tqdm
     pip install tqdm
     ```

## Usage

### Basic Usage
```bash
python transcription_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### With Custom Output Directory
```bash
python transcription_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID" --output-dir /path/to/output
```

## Output Structure

The pipeline creates a timestamped directory for each processed video with the following structure:

```
output/
└── video_YYYYMMDD_HHMMSS/
    ├── original_video_title.wav     # Original audio file
    ├── segment_0001_0.00-30.00.wav # Segmented audio files
    ├── segment_0002_30.00-60.00.wav
    ├── ...
    ├── transcriptions.csv          # Transcription results
    └── transcription_pipeline.log  # Process log
```

### Transcriptions CSV Format

The `transcriptions.csv` file contains the following columns:
- `file_path`: Path to the audio segment file
- `start_time`: Start time of the segment in seconds
- `end_time`: End time of the segment in seconds
- `text`: Transcribed text for the segment

## Pipeline Components

1. **YouTube Audio Download**
   - Downloads the best quality audio from YouTube videos
   - Automatically converts to WAV format
   - Handles various YouTube URL formats

2. **Audio Segmentation**
   - Uses Silero VAD for voice activity detection
   - Segments audio based on speech detection
   - Configurable parameters for segment length and silence detection
   - Skips segments longer than 120 seconds

3. **Speech Recognition**
   - Uses NVIDIA's Canary-1B model for transcription
   - Processes segments in batches for efficiency
   - Provides accurate timestamps for each transcribed segment

## Configuration Parameters

Key configuration parameters can be modified in the code:

- VAD Configuration:
  - `threshold`: 0.5 (speech detection threshold)
  - `min_speech_duration_ms`: 0
  - `max_speech_duration_s`: 30
  - `min_silence_duration_ms`: 2000
  - `speech_pad_ms`: 400

- Transcription Configuration:
  - `batch_size`: 16 (number of segments to process at once)
  - Maximum segment duration: 120 seconds

\
