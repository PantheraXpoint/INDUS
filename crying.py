import torch
import torchaudio
import numpy as np
import subprocess
from panns_inference import AudioTagging
import json

# ==== AUDIO EXTRACTION ====
def extract_audio_from_mp4(mp4_path, wav_path="temp_audio.wav"):
    """
    Extract audio from MP4 using ffmpeg (requires ffmpeg installed).
    Output: mono, 32kHz WAV file.
    """
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn", "-ac", "1", "-ar", "32000",
        "-acodec", "pcm_s16le", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

# ==== PREDICTION ====
def predict_audio_chunks(file_path, target_labels=["cry", "thud"], chunk_sec=3, topk=10):
    waveform, sr = torchaudio.load(file_path)
    if sr != 32000:
        waveform = torchaudio.functional.resample(waveform, sr, 32000)
    waveform = waveform.mean(dim=0).numpy()   # mono

    samples_per_chunk = chunk_sec * 32000
    total_samples = waveform.shape[0]
    num_chunks = int(np.ceil(total_samples / samples_per_chunk))

    results = []
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = min((i + 1) * samples_per_chunk, total_samples)
        chunk = waveform[start:end]

        if len(chunk) < 1000:  # skip if too short
            continue

        chunk = chunk[None, :]   # (1, samples)

        # Run inference
        clipwise_output, _ = at.inference(chunk)
        scores = clipwise_output[0]  # (527,)

        # Get top-k predictions
        indices = np.argsort(scores)[::-1][:topk]

        detected = []
        for idx in indices:
            label = at.labels[idx]
            score = scores[idx]
            for target in target_labels:
                if target.lower() in label.lower():
                    detected.append((label, float(score)))

        # Store results with time window
        if detected:
            results.append({
                "chunk_start_sec": start / 32000,
                "chunk_end_sec": end / 32000,
                "detections": detected
            })

    return results

if __name__ == "__main__":
    # ==== MODEL SETUP ====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    at = AudioTagging(checkpoint_path=None, device=device)  # CNN14 pretrained
    labels = at.labels  # 527 AudioSet class names
    # ==== DEMO USAGE ====
    mp4_file = "../Project-AVAS/datas/single_video/child_fall_1.mp4"
    wav_file = extract_audio_from_mp4(mp4_file)

    detections = predict_audio_chunks(wav_file)

    if detections:
        print("⚠️ Audio anomalies detected:")
        for d in detections:
            print(f"[{d['chunk_start_sec']:.1f}s → {d['chunk_end_sec']:.1f}s]")
            for label, score in d["detections"]:
                print(f"   - {label} (confidence {score:.2f})")
        output_json = {
            "detections": [
                {
                    "start_time": d["chunk_start_sec"],
                    "end_time": d["chunk_end_sec"],
                    "events": [
                        {"label": label, "confidence": score}
                        for label, score in d["detections"]
                    ]
                }
                for d in detections
            ]
        }
        with open("detections.json", "w") as f:
            json.dump(output_json, f, indent=2)
        print("💾 Detection results saved to detections.json")
    else:
        print("✅ No crying or thud detected.")
