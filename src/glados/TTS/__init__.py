"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


class PiperSpeechSynthesizer:
    """TTS synthesizer that calls a local Piper HTTP server."""

    def __init__(self, url: str = "http://127.0.0.1:5050"):
        self.url = url.rstrip("/")
        self.sample_rate = 44100  # will be updated after first request if needed

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        import io
        import json
        import wave
        import urllib.request
        import urllib.error

        print(f"[piper client] requesting speech for: {text!r}")

        payload = json.dumps({"input": text, "voice": "thorsten"}).encode()
        req = urllib.request.Request(
            f"{self.url}/v1/audio/speech",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            print(f"[piper client] ERROR connecting to piper server: {e}")
            return np.zeros(1, dtype=np.float32)

        print(f"[piper client] received {len(raw)} bytes")

        if len(raw) == 0:
            print("[piper client] ERROR: received empty response from server")
            return np.zeros(1, dtype=np.float32)

        try:
            buf = io.BytesIO(raw)
            with wave.open(buf, "rb") as wav:
                self.sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                frames = wav.readframes(n_frames)
                print(f"[piper client] WAV: {self.sample_rate} Hz, {n_frames} frames, {len(frames)} bytes of PCM")
        except Exception as e:
            print(f"[piper client] ERROR parsing WAV response: {e}")
            return np.zeros(1, dtype=np.float32)

        if len(frames) == 0:
            print("[piper client] ERROR: WAV file contained no audio frames")
            return np.zeros(1, dtype=np.float32)

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        print(f"[piper client] decoded {len(audio)} samples")
        return audio


# Factory function
def get_speech_synthesizer(
    voice: str = "glados",
) -> SpeechSynthesizerProtocol:
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.
    Parameters:
        voice (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer
            - "piper": Piper TTS server (German, local HTTP server on port 5050)
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() == "glados":
        from ..TTS import tts_glados
        return tts_glados.SpeechSynthesizer()

    if voice.lower() == "piper":
        return PiperSpeechSynthesizer(url="http://127.0.0.1:5050")

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer", "PiperSpeechSynthesizer"]