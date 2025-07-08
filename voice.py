# 系統需求分析與設計：智慧型語音轉錄器
# 功能：
# 1. 給一個影片（任意語言）
# 2. 抽取音訊（轉為 wav 格式）
# 3. 自動語言辨識 + 語音轉文字（Whisper）
# 4. 自動分辨不同講話人（Resemblyzer + KMeans）
# 5. 產生含時間戳的逐字稿 + 話者標註 + 輸出 JSON

import os
import tempfile
import json
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import whisper
import librosa
import soundfile as sf

def extract_audio(input_path: str, output_path: str = None) -> str:
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")
    clip = VideoFileClip(input_path)
    clip.audio.write_audiofile(output_path, fps=16000, nbytes=2, codec="pcm_s16le")
    return output_path


def estimate_num_speakers(embeddings: np.ndarray, max_speakers: int = 6) -> int:
    best_score, best_k = -1, 2
    for k in range(2, min(max_speakers, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def speaker_diarization(audio_path: str, window_size_sec: float = 1.5, num_speakers: int = None):
    wav, sr = librosa.load(audio_path, sr=16000)
    encoder = VoiceEncoder()
    slices, timestamps = [], []
    for i in range(0, len(wav), int(sr * window_size_sec)):
        chunk = wav[i:i + int(sr * window_size_sec)]
        if len(chunk) == int(sr * window_size_sec):
            slices.append(chunk)
            timestamps.append(i / sr)
    embeddings = np.array([encoder.embed_utterance(chunk) for chunk in slices])

    if num_speakers is None:
        num_speakers = estimate_num_speakers(embeddings)
        print(f"🔍 自動推估說話人數：{num_speakers}")

    kmeans = KMeans(n_clusters=num_speakers, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    segments = [
        {"start": round(timestamps[i], 2), "end": round(timestamps[i] + window_size_sec, 2), "speaker": f"Speaker {label}"}
        for i, label in enumerate(labels)
    ]
    return segments


def merge_transcript_with_speakers(transcript_segments, speaker_segments):
    result = []
    for seg in transcript_segments:
        ts_start = seg["start"]
        ts_end = seg["end"]
        matched = next((s for s in speaker_segments if s["start"] <= ts_start <= s["end"]), None)
        speaker = matched["speaker"] if matched else "Unknown"
        result.append({
            "start": ts_start,
            "end": ts_end,
            "speaker": speaker,
            "text": seg["text"].strip()
        })
    return result


def analyze_video(input_path: str, num_speakers: int = None, output_json: str = "transcript.json"):
    print("🎞️ Extracting audio from video...")
    audio_path = extract_audio(input_path)

    print("🧠 Loading Whisper and transcribing...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, task="transcribe")  # 自動語言辨識

    print("🧍 Running speaker diarization...")
    speaker_segments = speaker_diarization(audio_path, num_speakers=num_speakers)

    print("📜 Merging transcription and speakers...")
    merged = merge_transcript_with_speakers(result["segments"], speaker_segments)

    for entry in merged:
        sm, ss = divmod(entry["start"], 60)
        em, es = divmod(entry["end"], 60)
        print(f"[{int(sm):02d}:{int(ss):02d}-{int(em):02d}:{int(es):02d}] {entry['speaker']}: {entry['text']}")

    print(f"💾 Saving result to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return merged


# ✅ 使用範例：
if __name__ == "__main__":
    video_path = "sample_video.mp4"
    analyze_video(video_path, num_speakers=None, output_json="output.json")