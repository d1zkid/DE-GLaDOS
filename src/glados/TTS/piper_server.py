#!/usr/bin/env python3
"""Minimal Piper TTS server with audio post-processing."""

import argparse
import io
import json
import subprocess
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
from scipy import signal

# Find project root by searching upwards from __file__
def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "models").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to 4 parents (src/glados/TTS/piper_server.py -> project_root)
    return Path(__file__).resolve().parents[4]

def parse_args():
    parser = argparse.ArgumentParser(description="Piper TTS Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5050, help="Port to bind to")
    parser.add_argument("--model", default=None, help="Path to Piper model ONNX file")
    parser.add_argument("--piper-bin", default="piper", help="Path/name of piper binary")
    parser.add_argument("--length-scale", default="1.0", help="Speech length scale")
    parser.add_argument("--noise-scale", default="0.667", help="Noise scale")
    parser.add_argument("--noise-w-scale", default="0.95", help="Noise width scale")
    # Use parse_known_args in case of extra flags passed
    args, _ = parser.parse_known_args()
    return args

args = parse_args()

HOST = args.host
PORT = args.port
PIPER_BIN = args.piper_bin
LENGTH_SCALE = args.length_scale
NOISE_SCALE = args.noise_scale
NOISE_W_SCALE = args.noise_w_scale

if args.model:
    MODEL_PATH = Path(args.model)
else:
    MODEL_PATH = get_project_root() / "models/TTS/de-glados_1834-medium.onnx"

# --- Pronunciation fixes ---
# Words espeak-ng mispronounces. Add entries as you find them.
# Use espeak-ng inline phoneme notation: [[phonemes]]
# To find what a word currently produces: espeak-ng -v de -x "yourword"
PRONUNCIATION_FIXES: dict[str, str] = {
    "GLaDOS":   "[[glˈeɪdɔs]]",
    "Apertur":  "[[apɛʁˈtuːɐ̯]]",
    "Wheatley": "[[wiːtliː]]",
    "Chell":    "[[tʃɛl]]",
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Piper model not found at {MODEL_PATH}")

config_path = MODEL_PATH.with_suffix(".onnx.json")
with open(config_path) as f:
    _config = json.load(f)
SAMPLE_RATE = _config["audio"]["sample_rate"]

print(f"Sample rate: {SAMPLE_RATE} Hz")


def preprocess_text(text: str) -> str:
    for word, phonemes in PRONUNCIATION_FIXES.items():
        text = text.replace(word, phonemes)
    return text


def enhance_audio(pcm_bytes: bytes, original_rate: int) -> bytes:
    target_rate = 44100
    # Convert to float and normalize
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 1. High-Quality Polyphase Upsampling
    gcd = np.gcd(original_rate, target_rate)
    audio_44k = signal.resample_poly(audio, target_rate // gcd, original_rate // gcd)

    # 2. Spectral Band Replication (The "Air" Generator)
    sos_source = signal.butter(4, [4000, 10000], btype='bandpass', fs=target_rate, output='sos')
    source_band = signal.sosfilt(sos_source, audio_44k)

    t = np.arange(len(audio_44k)) / target_rate
    carrier = np.cos(2 * np.pi * 11025 * t)
    shifted_air = source_band * carrier

    sos_air_filter = signal.butter(6, 11000, btype='highpass', fs=target_rate, output='sos')
    air = signal.sosfilt(sos_air_filter, shifted_air)

    # 3. Dynamic Noise Gate
    envelope = np.abs(signal.hilbert(audio_44k))
    envelope = signal.convolve(envelope, np.ones(500)/500, mode='same')
    gate = np.where(envelope > 0.01, 1.0, 0.0)
    air = air * gate

    # 4. Final Mix & Presence Boost
    sos_tilt = signal.butter(2, 3000, btype='highpass', fs=target_rate, output='sos')
    presence = signal.sosfilt(sos_tilt, audio_44k)

    enhanced = audio_44k + (air * 0.6) + (presence * 0.15)

    # 5. Peak Limiting
    enhanced = np.clip(enhanced, -0.95, 0.95)

    return (enhanced * 32767).astype(np.int16).tobytes()


class PiperHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        text = body.get("input", "")

        text = preprocess_text(text)
        print(f"[piper] synthesizing: {text!r}")

        try:
            result = subprocess.run(
                [
                    PIPER_BIN,
                    "--model", str(MODEL_PATH),
                    "--output-raw",
                    "--length-scale", LENGTH_SCALE,
                    "--noise-scale", NOISE_SCALE,
                    "--noise-w-scale", NOISE_W_SCALE,
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            print("[piper] ERROR: timed out")
            self.send_error(500)
            return
        except FileNotFoundError:
            print(f"[piper] ERROR: piper binary not found")
            self.send_error(500)
            return

        if result.returncode != 0:
            print(f"[piper] ERROR: {result.stderr.decode()}")
            self.send_error(500)
            return

        raw_pcm = result.stdout
        print(f"[piper] raw PCM: {len(raw_pcm)} bytes")

        enhanced_pcm = enhance_audio(raw_pcm, SAMPLE_RATE)

        # WAV header must use target_rate (44100), not the model's original rate
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(enhanced_pcm)
        audio = buf.getvalue()

        print(f"[piper] sending {len(audio)} bytes")

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), PiperHandler)
    print(f"Piper server listening on {HOST}:{PORT}")
    server.serve_forever()
