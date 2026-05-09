import os
import glob
from pathlib import Path
import torch
import argparse
from faster_whisper import WhisperModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

MODEL_MAP = {
    1: "tiny",
    2: "base",
    3: "small",
    4: "medium",
    5: "large-v1",
    6: "large-v2",
    7: "large-v3",
}


def get_video_files(directory="./audio-or-video-files-to-transcribe"):
    video_extensions = ["*.mp4", "*.mkv", "*.webm", "*.flv", "*.mp3", "*.wav", "*.m4a"]
    files = []
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files


def transcribe_video_to_text(model, video_path, language):
    """Transcribe video/audio and return plain text."""
    print(f"  → Language: {language}")

    segments, _ = model.transcribe(
        video_path,
        language=language,
        task="transcribe",
        vad_filter=True,
    )

    return "\n".join(segment.text.strip() for segment in segments)


def main(languages, directory="./e", model_choice=4, overwrite=False):
    model_size = MODEL_MAP.get(model_choice, "medium")

    video_files = get_video_files(directory)
    if not video_files:
        print(f"No video/audio files found in {directory}.")
        return

    print(f"Found {len(video_files)} file(s).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    for video_path in video_files:
        video_stem = Path(video_path).stem
        print(f"\nProcessing: {video_path}")

        for lang in languages:
            txt_path = os.path.join(directory, f"{video_stem}_{lang}.txt")

            if os.path.exists(txt_path) and not overwrite:
                print(f"  Skipping {lang} (TXT already exists)")
                continue

            text_content = transcribe_video_to_text(model, video_path, lang)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            print(f"  Saved → {txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe videos/audio to plain text (multi-language)"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="./audio-or-video-files-to-transcribe",
        help="Directory containing video/audio files (default: ./audio-or-video-files-to-transcribe)",
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=4,
        help="Model size: 1=tiny, 2=base, 3=small, 4=medium, 5=large-v1, 6=large-v2, 7=large-v3",
    )
    parser.add_argument(
        "--overwrite-existing-txt-files",
        action="store_true",
        help="Overwrite existing .txt files",
    )

    args = parser.parse_args()

    # 👇 Change languages here
    languages = ["en"]
    main(languages, args.directory, args.model, args.overwrite_existing_txt_files)
