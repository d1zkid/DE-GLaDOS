"""Text-to-Speech (TTS) synthesis components.

This module provides a protocol-based interface for text-to-speech synthesis
and a factory function to create synthesizer instances for different voices.

Classes:
    SpeechSynthesizerProtocol: Protocol defining the TTS interface

Functions:
    get_speech_synthesizer: Factory function to create TTS instances
"""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class SpeechSynthesizerProtocol(Protocol):
    sample_rate: int

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]: ...


class PiperSpeechSynthesizer:
    """TTS synthesizer that calls a local Piper HTTP server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5050,
        model_path: str | None = None,
        piper_bin: str = "piper",
        length_scale: str = "1.0",
        noise_scale: str = "0.667",
        noise_w_scale: str = "0.95",
    ):
        self.url = f"http://{host}:{port}"
        self.sample_rate = 44100  # will be updated after first request if needed
        self.process = None

        self._start_server_if_needed(
            host=host,
            port=port,
            model_path=model_path,
            piper_bin=piper_bin,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

    def _start_server_if_needed(
        self,
        host: str,
        port: int,
        model_path: str | None,
        piper_bin: str,
        length_scale: str,
        noise_scale: str,
        noise_w_scale: str,
    ) -> None:
        import socket
        import subprocess
        import sys
        import time
        from loguru import logger
        from ..utils.resources import resource_path

        # Check if the port is already in use
        def is_port_in_use() -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex((host, port)) == 0

        if is_port_in_use():
            logger.info(f"Port {port} is already in use. Assuming Piper server is already running.")
            return

        # Locate the moved piper_server.py script in the same directory
        import os
        server_script = Path(__file__).parent / "piper_server.py"

        if not server_script.exists():
            logger.error(f"Piper server script not found at {server_script}")
            return

        # Build execution arguments
        cmd = [
            sys.executable,
            str(server_script),
            "--host", host,
            "--port", str(port),
            "--piper-bin", piper_bin,
            "--length-scale", length_scale,
            "--noise-scale", noise_scale,
            "--noise-w-scale", noise_w_scale,
        ]
        if model_path:
            # Resolve model_path relative to package root if needed
            resolved_model = resource_path(model_path)
            cmd.extend(["--model", str(resolved_model)])

        logger.info(f"Starting Piper server subprocess on {host}:{port} using {server_script}...")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"Failed to start Piper server subprocess: {e}")
            return

        # Wait for port to become active
        start_time = time.time()
        timeout = 10.0
        while time.time() - start_time < timeout:
            if is_port_in_use():
                logger.success("Piper server started successfully and is listening.")
                return
            if self.process.poll() is not None:
                logger.error(f"Piper server subprocess exited early with code {self.process.returncode}")
                return
            time.sleep(0.1)

        logger.warning("Piper server started but timed out waiting to listen on port.")

    def shutdown(self) -> None:
        import subprocess
        from loguru import logger
        if self.process:
            logger.info("Stopping Piper server subprocess...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
                logger.success("Piper server subprocess stopped.")
            except subprocess.TimeoutExpired:
                logger.warning("Piper server subprocess did not exit, killing...")
                self.process.kill()
                self.process.wait()
                logger.success("Piper server subprocess killed.")
            self.process = None

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
    piper_config: dict[str, Any] | None = None,
) -> SpeechSynthesizerProtocol:
    """
    Factory function to get an instance of an audio synthesizer based on the specified voice type.
    Parameters:
        voice (str): The type of TTS engine to use:
            - "glados": GLaDOS voice synthesizer
            - "piper": Piper TTS server (German, local HTTP server on port 5050)
            - <str>: Kokoro voice synthesizer using the specified voice <str> is available
        piper_config (dict[str, Any] | None): Optional configuration dictionary for Piper TTS.
    Returns:
        SpeechSynthesizerProtocol: An instance of the requested speech synthesizer
    Raises:
        ValueError: If the specified TTS engine type is not supported
    """
    if voice.lower() == "glados":
        from ..TTS import tts_glados
        return tts_glados.SpeechSynthesizer()

    if voice.lower() == "piper":
        cfg = piper_config or {}
        return PiperSpeechSynthesizer(
            host=cfg.get("host", "127.0.0.1"),
            port=cfg.get("port", 5050),
            model_path=cfg.get("model_path"),
            piper_bin=cfg.get("piper_bin", "piper"),
            length_scale=cfg.get("length_scale", "1.0"),
            noise_scale=cfg.get("noise_scale", "0.667"),
            noise_w_scale=cfg.get("noise_w_scale", "0.95"),
        )

    from ..TTS import tts_kokoro

    available_voices = tts_kokoro.get_voices()
    if voice not in available_voices:
        raise ValueError(f"Voice '{voice}' not available. Available voices: {available_voices}")

    return tts_kokoro.SpeechSynthesizer(voice=voice)


__all__ = ["SpeechSynthesizerProtocol", "get_speech_synthesizer", "PiperSpeechSynthesizer"]