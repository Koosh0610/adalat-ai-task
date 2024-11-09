import argparse
import os
import logging
import torch
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Import components
import yt_dlp
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from pydub import AudioSegment
from nemo.collections.asr.models import ASRModel
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

class TranscriptionPipeline:
    def __init__(self, output_base_dir: str = "output"):
        """Initialize the transcription pipeline."""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.output_base_dir = output_base_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize components
        self.init_vad_model()
        self.init_asr_model()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('transcription_pipeline.log')
            ]
        )

    def init_vad_model(self):
        """Initialize the Voice Activity Detection model."""
        try:
            self.logger.info("Loading VAD model...")
            self.vad_model = load_silero_vad()
            self.logger.info("VAD model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load VAD model: {str(e)}")
            raise

    def init_asr_model(self):
        """Initialize the ASR model."""
        try:
            self.logger.info("Loading ASR model...")
            self.asr_model = ASRModel.from_pretrained(model_name="nvidia/canary-1b")
            decode_cfg = self.asr_model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            self.asr_model.change_decoding_strategy(decode_cfg)
            self.asr_model.cuda()
            self.logger.info("ASR model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load ASR model: {str(e)}")
            raise

    def download_youtube_audio(self, url: str, output_dir: str) -> str:
        """Download audio from YouTube URL."""
        self.logger.info(f"Downloading audio from: {url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'quiet': True,
            'no_warnings': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'output').replace('/', '-')
                output_path = os.path.join(output_dir, f"{video_title}.wav")
                
                self.logger.info(f"Downloading: {video_title}")
                ydl.download([url])
                
                self.logger.info(f"Audio downloaded successfully")
                return output_path
        except Exception as e:
            self.logger.error(f"Failed to download audio: {str(e)}")
            raise

    def segment_audio(self, audio_path: str, output_dir: str) -> List[str]:
        """Segment audio file into chunks using VAD."""
        self.logger.info(f"Segmenting audio file: {audio_path}")
        
        try:
            # Read audio
            wav = read_audio(audio_path)
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                return_seconds=True,
                threshold=0.5,
                neg_threshold=0.35,
                min_speech_duration_ms=0,
                max_speech_duration_s=30,
                min_silence_duration_ms=2000,
                speech_pad_ms=400
            )

            # Load audio for segmentation
            audio = AudioSegment.from_wav(audio_path)
            
            # Process segments
            segment_files = []
            for i, interval in enumerate(speech_timestamps):
                start_ms = int(interval['start'] * 1000)
                end_ms = int(interval['end'] * 1000)
                
                # Skip segments that are too long
                duration = (end_ms - start_ms) / 1000
                if duration > 120:  # Skip segments longer than 120 seconds
                    continue
                
                segment = audio[start_ms:end_ms]
                output_file = os.path.join(
                    output_dir,
                    f"segment_{i+1:04d}_{start_ms/1000:.2f}-{end_ms/1000:.2f}.wav"
                )
                
                segment.export(output_file, format="wav")
                segment_files.append(output_file)
            
            self.logger.info(f"Created {len(segment_files)} segments")
            return segment_files
        
        except Exception as e:
            self.logger.error(f"Failed to segment audio: {str(e)}")
            raise

    def transcribe_segments(self, segment_files: List[str], output_dir: str) -> List[Dict]:
        """Transcribe audio segments."""
        self.logger.info(f"Transcribing {len(segment_files)} segments")
        
        transcriptions = []
        batch_size = 16
        
        try:
            for i in range(0, len(segment_files), batch_size):
                batch_files = segment_files[i:i + batch_size]
                self.logger.info(f"Transcribing batch {i // batch_size + 1}/{(len(segment_files) + batch_size - 1) // batch_size}")
                
                batch_transcriptions = self.asr_model.transcribe(audio=batch_files, batch_size=batch_size)
                
                for file_path, text in zip(batch_files, batch_transcriptions):
                    transcriptions.append({
                        'file_path': file_path,
                        'start_time': float(file_path.split('_')[-1].split('-')[0]),
                        'end_time': float(file_path.split('_')[-1].split('-')[1].replace('.wav', '')),
                        'text': text
                    })
            
            # Save transcriptions
            self._save_transcriptions(transcriptions, output_dir)
            return transcriptions
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe segments: {str(e)}")
            raise

    def _save_transcriptions(self, transcriptions: List[Dict], output_dir: str):
        """Save transcriptions to CSV file."""
        output_file = os.path.join(output_dir, "transcriptions.csv")
        
        try:
            with open(output_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['file_path', 'start_time', 'end_time', 'text'])
                writer.writeheader()
                writer.writerows(transcriptions)
            self.logger.info(f"Transcriptions saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save transcriptions: {str(e)}")
            raise

    def process_video(self, url: str) -> str:
        """Process a YouTube video through the entire pipeline."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_dir = os.path.join(self.output_base_dir, f"video_{timestamp}")
        os.makedirs(video_dir, exist_ok=True)
        
        try:
            # Download audio
            self.logger.info("Step 1: Downloading audio...")
            audio_path = self.download_youtube_audio(url, video_dir)
            
            # Segment audio
            self.logger.info("Step 2: Segmenting audio...")
            segment_files = self.segment_audio(audio_path, video_dir)
            
            # Transcribe segments
            self.logger.info("Step 3: Transcribing segments...")
            transcriptions = self.transcribe_segments(segment_files, video_dir)
            
            self.logger.info(f"Pipeline completed successfully. Results saved in {video_dir}")
            return video_dir
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="YouTube Video Transcription Pipeline")
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument("--output-dir", type=str, default="output", help="Base output directory")
    
    args = parser.parse_args()
    
    try:
        pipeline = TranscriptionPipeline(output_base_dir=args.output_dir)
        output_dir = pipeline.process_video(args.url)
        print(f"Processing completed successfully. Results saved in: {output_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())