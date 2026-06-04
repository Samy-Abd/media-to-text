from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

FORMAT_SUFFIXES = {
    "plain": ".plain.txt",
    "txt": ".txt",
    "json": ".json",
    "srt": ".srt",
    "vtt": ".vtt",
}

DEFAULT_INITIAL_PROMPT = (
    "The recording may contain French, English, or both. "
    "Transcribe exactly what is spoken, preserving the spoken language."
)

CHUNK_MANIFEST_NAME = "chunks.json"


def parse_csv_set(value: str, *, valid_values: set[str] | None = None) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("value cannot be empty")

    if valid_values is not None:
        unknown = sorted(set(items) - valid_values)
        if unknown:
            raise argparse.ArgumentTypeError(
                f"unknown value(s): {', '.join(unknown)}. Valid values: {', '.join(sorted(valid_values))}"
            )

    return items


def parse_extensions(value: str) -> set[str]:
    extensions: set[str] = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "default":
            extensions.update(MEDIA_EXTENSIONS)
            continue
        if not item.startswith("."):
            item = f".{item}"
        extensions.add(item)

    if not extensions:
        raise argparse.ArgumentTypeError("at least one extension is required")
    return extensions


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_audio_files(
    input_dir: Path,
    extensions: set[str],
    excluded_dirs: Iterable[Path] = (),
) -> list[Path]:
    excluded_roots = [path.resolve() for path in excluded_dirs]
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and not any(is_relative_to(path.resolve(), root) for root in excluded_roots)
    )


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def resolve_device_and_compute(device: str, compute_type: str) -> tuple[str, str]:
    resolved_device = device
    if device == "auto":
        resolved_device = "cpu"
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                resolved_device = "cuda"
        except Exception:
            resolved_device = "cpu"

    resolved_compute_type = compute_type
    if compute_type == "auto":
        resolved_compute_type = "float16" if resolved_device == "cuda" else "int8"

    return resolved_device, resolved_compute_type


def import_faster_whisper() -> tuple[Any, Any]:
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        return WhisperModel, BatchedInferencePipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Install it with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def import_av() -> Any:
    try:
        import av

        return av
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: PyAV. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def output_paths_for(
    audio_path: Path,
    input_dir: Path,
    output_dir: Path,
    formats: Iterable[str],
) -> dict[str, Path]:
    relative = audio_path.relative_to(input_dir)
    base_dir = output_dir / relative.parent
    return {
        output_format: base_dir / f"{relative.stem}{FORMAT_SUFFIXES[output_format]}"
        for output_format in formats
    }


def resolve_extracted_audio_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.extracted_audio_dir is not None:
        return args.extracted_audio_dir.expanduser().resolve()
    return output_dir / "_extracted_audio"


def resolve_chunk_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.chunk_dir is not None:
        return args.chunk_dir.expanduser().resolve()
    return output_dir / "_chunks"


def extracted_audio_path_for(
    media_path: Path,
    input_dir: Path,
    extracted_audio_dir: Path,
) -> Path:
    relative = media_path.relative_to(input_dir)
    return extracted_audio_dir / relative.parent / f"{relative.stem}.wav"


def chunk_dir_for(source_path: Path, input_dir: Path, chunk_root: Path) -> Path:
    relative = source_path.relative_to(input_dir)
    return chunk_root / relative.parent / relative.stem


def should_skip_output(paths: dict[str, Path], overwrite: bool) -> bool:
    return not overwrite and all(path.exists() for path in paths.values())


def timestamp(seconds: float | None, *, decimal_marker: str = ".", always_hours: bool = True) -> str:
    if seconds is None:
        seconds = 0.0
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)

    if always_hours or hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"
    return f"{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


def segment_to_dict(segment: Any) -> dict[str, Any]:
    words = getattr(segment, "words", None)
    data: dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "seek": getattr(segment, "seek", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": getattr(segment, "text", "").strip(),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "temperature": getattr(segment, "temperature", None),
    }
    if words:
        data["words"] = [
            {
                "start": getattr(word, "start", None),
                "end": getattr(word, "end", None),
                "word": getattr(word, "word", ""),
                "probability": getattr(word, "probability", None),
            }
            for word in words
        ]
    return data


def language_probs(info: Any) -> list[dict[str, Any]] | None:
    probs = getattr(info, "all_language_probs", None)
    if not probs:
        return None
    return [{"language": language, "probability": probability} for language, probability in probs]


def build_metadata(
    source_path: Path,
    transcription_inputs: list[dict[str, Any]],
    info_summary: dict[str, Any],
    chunk_infos: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
    compute_type: str,
    effective_batch_size: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    transcription_input_paths = [str(chunk["path"]) for chunk in transcription_inputs]
    return {
        "source": str(source_path),
        "transcription_input": transcription_input_paths[0]
        if len(transcription_input_paths) == 1
        else None,
        "transcription_inputs": transcription_input_paths,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "model": {
            "model_size": args.model_size,
            "device": device,
            "compute_type": compute_type,
            "batch_size": effective_batch_size,
            "beam_size": args.beam_size,
        },
        "audio": {
            "duration": info_summary.get("duration"),
            "duration_after_vad": info_summary.get("duration_after_vad"),
        },
        "language": {
            "requested": None if args.language == "auto" else args.language,
            "detected": info_summary.get("language"),
            "probability": info_summary.get("language_probability"),
            "all_language_probs": None,
        },
        "chunks": chunk_infos,
        "options": {
            "task": args.task,
            "vad": args.vad,
            "vad_min_silence_ms": args.vad_min_silence_ms,
            "vad_speech_pad_ms": args.vad_speech_pad_ms,
            "extract_video_audio": args.extract_video_audio,
            "extracted_audio_sample_rate": args.extracted_audio_sample_rate,
            "chunk_seconds": args.chunk_seconds,
            "chunk_sample_rate": args.chunk_sample_rate,
            "word_timestamps": args.word_timestamps,
            "condition_on_previous_text": args.condition_on_previous_text,
            "retry_unbatched_on_oom": args.retry_unbatched_on_oom,
            "initial_prompt": None if args.no_initial_prompt else args.initial_prompt,
        },
    }


def write_plain(path: Path, segments: list[dict[str, Any]]) -> None:
    paragraphs: list[str] = []
    current: list[str] = []
    previous_end: float | None = None

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        start = segment.get("start")
        if previous_end is not None and start is not None and start - previous_end > 4.0 and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        previous_end = segment.get("end")

    if current:
        paragraphs.append(" ".join(current))

    path.write_text("\n\n".join(paragraphs).strip() + "\n", encoding="utf-8")


def write_txt(path: Path, metadata: dict[str, Any], segments: list[dict[str, Any]]) -> None:
    lines = [
        f"Source: {metadata['source']}",
        f"Created UTC: {metadata['created_at']}",
        f"Model: {metadata['model']['model_size']} ({metadata['model']['device']}, {metadata['model']['compute_type']})",
        (
            "Language: "
            f"{metadata['language']['detected']} "
            f"(probability: {metadata['language']['probability']})"
        ),
        "",
    ]
    for segment in segments:
        lines.append(
            f"[{timestamp(segment.get('start'))} - {timestamp(segment.get('end'))}] {segment['text']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, metadata: dict[str, Any], segments: list[dict[str, Any]]) -> None:
    payload = {"metadata": metadata, "segments": segments}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_srt(path: Path, segments: list[dict[str, Any]]) -> None:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment["text"].strip()
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{timestamp(segment.get('start'), decimal_marker=',')} --> "
                    f"{timestamp(segment.get('end'), decimal_marker=',')}",
                    text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks).rstrip() + "\n", encoding="utf-8")


def write_vtt(path: Path, segments: list[dict[str, Any]]) -> None:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    f"{timestamp(segment.get('start'))} --> {timestamp(segment.get('end'))}",
                    text,
                ]
            )
        )
        blocks.append("")
    path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")


def write_outputs(
    paths: dict[str, Path],
    metadata: dict[str, Any],
    segments: list[dict[str, Any]],
) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if "plain" in paths:
        write_plain(paths["plain"], segments)
    if "txt" in paths:
        write_txt(paths["txt"], metadata, segments)
    if "json" in paths:
        write_json(paths["json"], metadata, segments)
    if "srt" in paths:
        write_srt(paths["srt"], segments)
    if "vtt" in paths:
        write_vtt(paths["vtt"], segments)


def build_vad_parameters(args: argparse.Namespace) -> dict[str, int]:
    return {
        "min_silence_duration_ms": args.vad_min_silence_ms,
        "speech_pad_ms": args.vad_speech_pad_ms,
    }


def is_memory_allocation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "unable to allocate" in message
        or "out of memory" in message
        or "cuda out of memory" in message
    )


def media_duration_seconds(media_path: Path) -> float | None:
    av = import_av()
    container = None
    try:
        container = av.open(str(media_path))
        if container.duration is not None:
            return float(container.duration * av.time_base)

        audio_stream = next(iter(container.streams.audio), None)
        if (
            audio_stream is not None
            and audio_stream.duration is not None
            and audio_stream.time_base is not None
        ):
            return float(audio_stream.duration * audio_stream.time_base)
    finally:
        if container is not None:
            container.close()

    return None


def read_chunk_manifest(chunk_dir: Path) -> list[dict[str, Any]] | None:
    manifest_path = chunk_dir / CHUNK_MANIFEST_NAME
    if not manifest_path.exists():
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return None

    prepared_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        path = chunk_dir / chunk["file"]
        if not path.exists():
            return None
        prepared_chunks.append(
            {
                "path": path,
                "offset": float(chunk["offset"]),
                "duration": float(chunk.get("duration", 0.0)),
            }
        )
    return prepared_chunks


def write_chunk_manifest(
    chunk_dir: Path,
    source_path: Path,
    sample_rate: int,
    chunk_seconds: int,
    chunks: list[dict[str, Any]],
) -> None:
    payload = {
        "source": str(source_path),
        "sample_rate": sample_rate,
        "chunk_seconds": chunk_seconds,
        "chunks": [
            {
                "file": chunk["path"].name,
                "offset": chunk["offset"],
                "duration": chunk.get("duration"),
            }
            for chunk in chunks
        ],
    }
    (chunk_dir / CHUNK_MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def cleanup_chunk_dir(chunk_dir: Path) -> None:
    if not chunk_dir.exists():
        return
    for path in chunk_dir.glob("chunk_*.wav"):
        path.unlink()
    manifest_path = chunk_dir / CHUNK_MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()


def open_wav_writer(av: Any, path: Path, sample_rate: int) -> tuple[Any, Any]:
    container = av.open(str(path), mode="w", format="wav")
    stream = container.add_stream("pcm_s16le", rate=sample_rate)
    stream.layout = "mono"
    return container, stream


def close_wav_writer(container: Any, stream: Any) -> None:
    for output_packet in stream.encode():
        container.mux(output_packet)
    container.close()


def chunk_audio_for_transcription(
    media_path: Path,
    chunk_dir: Path,
    chunk_seconds: int,
    sample_rate: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if chunk_seconds <= 0:
        return [{"path": media_path, "offset": 0.0, "duration": None}]

    if not overwrite:
        existing_chunks = read_chunk_manifest(chunk_dir)
        if existing_chunks:
            print(f"Using existing audio chunks: {chunk_dir}")
            return existing_chunks

    av = import_av()
    chunk_dir.mkdir(parents=True, exist_ok=True)
    cleanup_chunk_dir(chunk_dir)

    input_container = None
    output_container = None
    output_stream = None
    chunks: list[dict[str, Any]] = []
    failed = False
    chunk_index = 0
    chunk_start_sample = 0
    current_chunk_samples = 0
    total_samples = 0
    chunk_samples = chunk_seconds * sample_rate

    try:
        input_container = av.open(str(media_path))
        audio_stream = next(iter(input_container.streams.audio), None)
        if audio_stream is None:
            raise RuntimeError("No audio stream found in media file.")

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        def start_chunk() -> None:
            nonlocal output_container, output_stream, chunk_index, chunk_start_sample
            chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.wav"
            output_container, output_stream = open_wav_writer(av, chunk_path, sample_rate)
            chunks.append(
                {
                    "path": chunk_path,
                    "offset": chunk_start_sample / sample_rate,
                    "duration": 0.0,
                }
            )

        def finish_chunk() -> None:
            nonlocal output_container, output_stream, current_chunk_samples, chunk_index
            nonlocal chunk_start_sample
            if output_container is None or output_stream is None:
                return
            close_wav_writer(output_container, output_stream)
            chunks[-1]["duration"] = current_chunk_samples / sample_rate
            chunk_start_sample += current_chunk_samples
            current_chunk_samples = 0
            chunk_index += 1
            output_container = None
            output_stream = None

        start_chunk()
        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                resampled_frames = resampler.resample(frame)
                if resampled_frames is None:
                    continue
                if not isinstance(resampled_frames, list):
                    resampled_frames = [resampled_frames]
                for resampled_frame in resampled_frames:
                    if current_chunk_samples >= chunk_samples:
                        finish_chunk()
                        start_chunk()
                    for output_packet in output_stream.encode(resampled_frame):
                        output_container.mux(output_packet)
                    current_chunk_samples += resampled_frame.samples
                    total_samples += resampled_frame.samples

        if current_chunk_samples == 0 and chunks:
            output_container.close()
            chunks.pop()
            output_container = None
            output_stream = None
        else:
            finish_chunk()

        if total_samples == 0:
            raise RuntimeError("The media audio stream did not produce any decodable frames.")

        write_chunk_manifest(chunk_dir, media_path, sample_rate, chunk_seconds, chunks)
    except Exception:
        failed = True
        cleanup_chunk_dir(chunk_dir)
        raise
    finally:
        if output_container is not None:
            output_container.close()
        if input_container is not None:
            input_container.close()
        if failed:
            cleanup_chunk_dir(chunk_dir)

    print(f"Created {len(chunks)} audio chunk(s): {chunk_dir}")
    return chunks


def extract_audio_from_video(
    video_path: Path,
    audio_path: Path,
    sample_rate: int,
    overwrite: bool,
) -> Path:
    if audio_path.exists() and not overwrite:
        print(f"Using existing extracted audio: {audio_path}")
        return audio_path

    av = import_av()
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = audio_path.with_name(f".{audio_path.name}.tmp.wav")
    if temp_path.exists():
        temp_path.unlink()

    input_container = None
    output_container = None
    decoded_frames = 0
    failed = False
    try:
        input_container = av.open(str(video_path))
        audio_stream = next(iter(input_container.streams.audio), None)
        if audio_stream is None:
            raise RuntimeError("No audio stream found in video file.")

        output_container = av.open(str(temp_path), mode="w", format="wav")
        output_stream = output_container.add_stream("pcm_s16le", rate=sample_rate)
        output_stream.layout = "mono"
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        for packet in input_container.demux(audio_stream):
            for frame in packet.decode():
                decoded_frames += 1
                resampled_frames = resampler.resample(frame)
                if resampled_frames is None:
                    continue
                if not isinstance(resampled_frames, list):
                    resampled_frames = [resampled_frames]
                for resampled_frame in resampled_frames:
                    for output_packet in output_stream.encode(resampled_frame):
                        output_container.mux(output_packet)

        for output_packet in output_stream.encode():
            output_container.mux(output_packet)

        if decoded_frames == 0:
            raise RuntimeError("The video audio stream did not produce any decodable frames.")
    except Exception:
        failed = True
        raise
    finally:
        if output_container is not None:
            output_container.close()
        if input_container is not None:
            input_container.close()
        if failed and temp_path.exists():
            temp_path.unlink()

    temp_path.replace(audio_path)
    print(f"Extracted audio: {audio_path}")
    return audio_path


def prepare_transcription_input(
    source_path: Path,
    input_dir: Path,
    extracted_audio_dir: Path,
    args: argparse.Namespace,
) -> Path:
    if not args.extract_video_audio or not is_video_file(source_path):
        return source_path

    audio_path = extracted_audio_path_for(source_path, input_dir, extracted_audio_dir)
    return extract_audio_from_video(
        source_path,
        audio_path,
        args.extracted_audio_sample_rate,
        args.overwrite,
    )


def prepare_transcription_inputs(
    source_path: Path,
    input_dir: Path,
    extracted_audio_dir: Path,
    chunk_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    transcription_input = prepare_transcription_input(
        source_path,
        input_dir,
        extracted_audio_dir,
        args,
    )

    if args.chunk_seconds <= 0:
        return [{"path": transcription_input, "offset": 0.0, "duration": None}]

    duration = media_duration_seconds(transcription_input)
    if duration is not None and duration <= args.chunk_seconds:
        return [{"path": transcription_input, "offset": 0.0, "duration": duration}]

    return chunk_audio_for_transcription(
        transcription_input,
        chunk_dir_for(source_path, input_dir, chunk_root),
        args.chunk_seconds,
        args.chunk_sample_rate,
        args.overwrite,
    )


def consume_segments(
    segments_iter: Iterable[Any],
    duration: float | None,
    show_progress: bool,
    label: str,
) -> list[dict[str, Any]]:
    progress_bar = None
    last_position = 0.0
    try:
        if show_progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(
                    total=duration,
                    unit="sec",
                    desc=label,
                    leave=False,
                    dynamic_ncols=True,
                )
            except ImportError:
                progress_bar = None

        segments: list[dict[str, Any]] = []
        for segment in segments_iter:
            segment_dict = segment_to_dict(segment)
            segments.append(segment_dict)
            end = segment_dict.get("end")
            if progress_bar is not None and end is not None:
                position = min(float(end), float(duration or end))
                progress_bar.update(max(0.0, position - last_position))
                last_position = position

        if progress_bar is not None and duration is not None:
            progress_bar.update(max(0.0, float(duration) - last_position))
        return segments
    finally:
        if progress_bar is not None:
            progress_bar.close()


def offset_segments(segments: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    if offset == 0:
        return segments

    for segment in segments:
        if segment.get("start") is not None:
            segment["start"] += offset
        if segment.get("end") is not None:
            segment["end"] += offset
        for word in segment.get("words", []):
            if word.get("start") is not None:
                word["start"] += offset
            if word.get("end") is not None:
                word["end"] += offset
    return segments


def summarize_chunk_infos(chunk_infos: list[dict[str, Any]]) -> dict[str, Any]:
    languages = [
        chunk_info.get("language")
        for chunk_info in chunk_infos
        if chunk_info.get("language")
    ]
    unique_languages = sorted(set(languages))
    if len(unique_languages) == 1:
        detected_language = unique_languages[0]
        language_probability = next(
            (
                chunk_info.get("language_probability")
                for chunk_info in chunk_infos
                if chunk_info.get("language") == detected_language
            ),
            None,
        )
    elif unique_languages:
        detected_language = "mixed"
        language_probability = None
    else:
        detected_language = None
        language_probability = None

    duration = 0.0
    duration_after_vad = 0.0
    has_duration = False
    has_vad_duration = False
    for chunk_info in chunk_infos:
        if chunk_info.get("duration") is not None:
            duration += float(chunk_info["duration"])
            has_duration = True
        if chunk_info.get("duration_after_vad") is not None:
            duration_after_vad += float(chunk_info["duration_after_vad"])
            has_vad_duration = True

    return {
        "language": detected_language,
        "language_probability": language_probability,
        "duration": duration if has_duration else None,
        "duration_after_vad": duration_after_vad if has_vad_duration else None,
    }


def transcribe_file(
    source_path: Path,
    transcription_inputs: list[dict[str, Any]],
    output_paths: dict[str, Path],
    transcriber: Any,
    args: argparse.Namespace,
    device: str,
    compute_type: str,
    use_batched_pipeline: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    language = None if args.language == "auto" else args.language
    initial_prompt = None if args.no_initial_prompt else args.initial_prompt

    transcribe_kwargs: dict[str, Any] = {
        "beam_size": args.beam_size,
        "language": language,
        "task": args.task,
        "vad_filter": args.vad,
        "vad_parameters": build_vad_parameters(args),
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
        "initial_prompt": initial_prompt,
    }
    if use_batched_pipeline:
        transcribe_kwargs["batch_size"] = args.batch_size

    segments: list[dict[str, Any]] = []
    chunk_infos: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(transcription_inputs, start=1):
        chunk_path = chunk["path"]
        chunk_offset = float(chunk.get("offset", 0.0))
        chunk_label = (
            source_path.name
            if len(transcription_inputs) == 1
            else f"{source_path.name} chunk {chunk_index}/{len(transcription_inputs)}"
        )
        segments_iter, info = transcriber.transcribe(str(chunk_path), **transcribe_kwargs)
        print(
            f"Detected {getattr(info, 'language', 'unknown')} "
            f"({getattr(info, 'language_probability', None)}) for {chunk_label}"
        )

        chunk_segments = consume_segments(
            segments_iter,
            getattr(info, "duration", None),
            args.progress,
            chunk_label,
        )
        segments.extend(offset_segments(chunk_segments, chunk_offset))
        chunk_infos.append(
            {
                "path": str(chunk_path),
                "offset": chunk_offset,
                "duration": getattr(info, "duration", chunk.get("duration")),
                "duration_after_vad": getattr(info, "duration_after_vad", None),
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
            }
        )

    elapsed_seconds = time.perf_counter() - start
    effective_batch_size = args.batch_size if use_batched_pipeline else 1
    info_summary = summarize_chunk_infos(chunk_infos)
    metadata = build_metadata(
        source_path,
        transcription_inputs,
        info_summary,
        chunk_infos,
        args,
        device,
        compute_type,
        effective_batch_size,
        elapsed_seconds,
    )
    write_outputs(output_paths, metadata, segments)
    return {
        "source": str(source_path),
        "transcription_inputs": [str(chunk["path"]) for chunk in transcription_inputs],
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "segments": len(segments),
        "elapsed_seconds": elapsed_seconds,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe a folder of long audio or video files with faster-whisper."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing audio or video files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("transcripts"),
        help="Folder where transcript files are written. Defaults to ./transcripts.",
    )
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="faster-whisper model name or local model path. Defaults to large-v3.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device. auto uses CUDA when CTranslate2 can see a GPU.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type. auto uses float16 on CUDA and int8 on CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batched decoding size. Lower this if GPU/CPU memory is tight.",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    parser.add_argument(
        "--language",
        choices=["auto", "fr", "en"],
        default="auto",
        help="Language hint. Use auto for French, English, or mixed recordings.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Use transcribe to preserve the spoken language; translate outputs English.",
    )
    parser.add_argument(
        "--formats",
        type=lambda value: parse_csv_set(value, valid_values=set(FORMAT_SUFFIXES)),
        default=parse_csv_set("txt", valid_values=set(FORMAT_SUFFIXES)),
        help="Comma-separated output formats: plain,txt,json,srt,vtt.",
    )
    parser.add_argument(
        "--extensions",
        type=parse_extensions,
        default=MEDIA_EXTENSIONS,
        help="Comma-separated media extensions, or 'default'.",
    )
    parser.add_argument(
        "--extract-video-audio",
        dest="extract_video_audio",
        action="store_true",
        default=True,
        help="Extract MP4/MOV/MKV/WebM audio to WAV before transcription.",
    )
    parser.add_argument(
        "--no-extract-video-audio",
        dest="extract_video_audio",
        action="store_false",
        help="Send video files directly to faster-whisper without saving extracted WAV audio.",
    )
    parser.add_argument(
        "--extracted-audio-dir",
        type=Path,
        default=None,
        help="Folder for extracted WAV files. Defaults to <output-dir>/_extracted_audio.",
    )
    parser.add_argument(
        "--extracted-audio-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for extracted mono WAV audio.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=1800,
        help="Split media longer than this into WAV chunks before transcription. Use 0 to disable.",
    )
    parser.add_argument(
        "--chunk-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for generated chunk WAV files.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Folder for generated audio chunks. Defaults to <output-dir>/_chunks.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word timestamps in JSON output. Slower but useful for alignment.",
    )
    parser.add_argument("--vad", dest="vad", action="store_true", default=True)
    parser.add_argument("--no-vad", dest="vad", action="store_false")
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=1000,
        help="Silero VAD silence threshold in milliseconds.",
    )
    parser.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=200,
        help="Silero VAD padding around speech in milliseconds.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        help="Enable Whisper context conditioning across segments.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Prompt used to bias punctuation/language handling.",
    )
    parser.add_argument(
        "--no-initial-prompt",
        action="store_true",
        help="Disable the default bilingual initial prompt.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU threads for CTranslate2. 0 lets the library choose.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Workers used by faster-whisper. Increase only when processing many files.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for downloaded model files.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached model files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe files even when all requested outputs already exist.",
    )
    parser.add_argument(
        "--retry-unbatched-on-oom",
        dest="retry_unbatched_on_oom",
        action="store_true",
        default=True,
        help="Retry a file with batch size 1 when batched inference runs out of memory.",
    )
    parser.add_argument(
        "--no-retry-unbatched-on-oom",
        dest="retry_unbatched_on_oom",
        action="store_false",
        help="Disable automatic unbatched retry after memory-allocation errors.",
    )
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without loading the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        parser.error(f"input_dir does not exist or is not a directory: {input_dir}")

    extracted_audio_dir = resolve_extracted_audio_dir(args, output_dir)
    chunk_root = resolve_chunk_dir(args, output_dir)
    excluded_dirs = [
        path
        for path in (output_dir, extracted_audio_dir, chunk_root)
        if path != input_dir and is_relative_to(path, input_dir)
    ]

    audio_files = discover_audio_files(input_dir, args.extensions, excluded_dirs)
    if not audio_files:
        print(f"No media files found in {input_dir}")
        return 0

    jobs: list[tuple[Path, dict[str, Path]]] = []
    skipped = 0
    for audio_path in audio_files:
        paths = output_paths_for(audio_path, input_dir, output_dir, args.formats)
        if should_skip_output(paths, args.overwrite):
            skipped += 1
            continue
        jobs.append((audio_path, paths))

    print(f"Found {len(audio_files)} media file(s); {len(jobs)} queued; {skipped} skipped.")
    if args.dry_run:
        for audio_path, paths in jobs:
            print(f"- {audio_path}")
            if args.extract_video_audio and is_video_file(audio_path):
                print(
                    f"  extracted_audio: "
                    f"{extracted_audio_path_for(audio_path, input_dir, extracted_audio_dir)}"
                )
            if args.chunk_seconds > 0:
                print(f"  chunks: {chunk_dir_for(audio_path, input_dir, chunk_root)}")
            for output_format, path in paths.items():
                print(f"  {output_format}: {path}")
        return 0

    device, compute_type = resolve_device_and_compute(args.device, args.compute_type)
    print(
        f"Loading model '{args.model_size}' on {device} with compute_type={compute_type}. "
        "The first run may download model files."
    )

    try:
        WhisperModel, BatchedInferencePipeline = import_faster_whisper()
        model_kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": args.cpu_threads,
            "num_workers": args.num_workers,
            "local_files_only": args.local_files_only,
        }
        if args.model_cache_dir is not None:
            model_kwargs["download_root"] = str(args.model_cache_dir.expanduser().resolve())

        model = WhisperModel(args.model_size, **model_kwargs)
        use_batched_pipeline = args.batch_size > 1
        primary_transcriber = (
            BatchedInferencePipeline(model=model) if use_batched_pipeline else model
        )
    except Exception as exc:
        print(f"Failed to initialize faster-whisper: {exc}", file=sys.stderr)
        return 1

    failures: list[tuple[Path, str]] = []
    successes: list[dict[str, Any]] = []
    for index, (audio_path, paths) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Transcribing {audio_path}")
        transcription_inputs = [{"path": audio_path, "offset": 0.0, "duration": None}]
        try:
            transcription_inputs = prepare_transcription_inputs(
                audio_path,
                input_dir,
                extracted_audio_dir,
                chunk_root,
                args,
            )
            successes.append(
                transcribe_file(
                    audio_path,
                    transcription_inputs,
                    paths,
                    primary_transcriber,
                    args,
                    device,
                    compute_type,
                    use_batched_pipeline,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if (
                args.retry_unbatched_on_oom
                and use_batched_pipeline
                and is_memory_allocation_error(exc)
            ):
                print(
                    "Batched inference ran out of memory; retrying this file "
                    "with batch_size=1 streaming mode."
                )
                try:
                    successes.append(
                        transcribe_file(
                            audio_path,
                            transcription_inputs,
                            paths,
                            model,
                            args,
                            device,
                            compute_type,
                            False,
                        )
                    )
                    continue
                except KeyboardInterrupt:
                    raise
                except Exception as retry_exc:
                    exc = retry_exc

            failures.append((audio_path, str(exc)))
            print(f"Failed: {audio_path}: {exc}", file=sys.stderr)

    print(f"Completed {len(successes)} file(s).")
    if failures:
        print(f"Failed {len(failures)} file(s):", file=sys.stderr)
        for audio_path, message in failures:
            print(f"- {audio_path}: {message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
