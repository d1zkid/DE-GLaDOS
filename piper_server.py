#!/usr/bin/env python3
"""Minimal Piper TTS server with audio post-processing."""

import io
import json
import subprocess
import wave
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from scipy import signal

MODEL_PATH = Path(__file__).parent / "models/TTS/de-glados_1834-medium.onnx"
HOST = "127.0.0.1"
PORT = 5050
PIPER_BIN = "piper"

# --- Inference tuning (won't break anything, just change delivery) ---
# length-scale: speaking rate. 0.9 = slightly snappier. Default: 1.0
# noise-scale:  phoneme timing variance. Default: 0.667
# noise-w-scale: pitch variance. Raise to 0.9-1.0 for more expression. Default: 0.8
LENGTH_SCALE   = "1"
NOISE_SCALE    = "0.667"
NOISE_W_SCALE  = "0.95"   # note: flag is --noise-w-scale, not --noise-scale-w

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
