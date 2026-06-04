# Long Audio STT Pipeline

Batch transcription pipeline for long audio and video files using `faster-whisper` and `large-v3`.

It accepts folders of `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`, `.mkv`, `.webm`, and similar media files, extracts audio when needed, splits long recordings into chunks, and writes a merged `.txt` transcript for each source file.

## Features

- Recursive folder processing
- MP4 and other video input support
- Automatic audio extraction to mono 16 kHz WAV
- Chunking for long recordings
- French, English, or mixed-language transcription
- CUDA and CPU support
- Resumable output skipping

## Requirements

- Python 3.9+
- `faster-whisper`
- `PyAV`

On NVIDIA GPUs, you also need working CUDA libraries compatible with CTranslate2.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

Transcribe a folder:

```powershell
python .\stt_pipeline.py ".\audio" -o ".\transcripts"
```

Process a folder of MP4 files:

```powershell
python .\stt_pipeline.py ".\videos" -o ".\transcripts"
```

Show what would be processed:

```powershell
python .\stt_pipeline.py ".\audio" --dry-run
```

Force CPU mode:

```powershell
python .\stt_pipeline.py ".\audio" --device cpu --batch-size 1
```

Use smaller chunks for very long files:

```powershell
python .\stt_pipeline.py ".\audio" --chunk-seconds 600
```

Disable chunking:

```powershell
python .\stt_pipeline.py ".\audio" --chunk-seconds 0
```

## Output

By default, the pipeline writes one `.txt` file per input media file.

Example:

```text
audio\meeting.mp4
transcripts\meeting.txt
```

Intermediate extracted audio is stored in:

```text
<output-dir>\_extracted_audio\
```

Chunk files are stored in:

```text
<output-dir>\_chunks\
```

## Options

- `--language auto|fr|en`
- `--task transcribe|translate`
- `--device auto|cuda|cpu`
- `--compute-type auto|float16|int8`
- `--batch-size N`
- `--chunk-seconds N`
- `--no-extract-video-audio`
- `--overwrite`

## Notes

- The default model is `large-v3`.
- The default output is `.txt` only.
- Long files are chunked before transcription, then merged back into a single transcript.
- If batched inference runs out of memory, the pipeline retries in unbatched mode.
