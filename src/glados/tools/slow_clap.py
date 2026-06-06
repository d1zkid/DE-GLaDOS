import queue
from typing import Any

from loguru import logger
import sounddevice as sd  # type: ignore
import soundfile as sf

tool_definition = {
    "type": "function",
    "function": {
        "name": "slow clap",
        "description": "Führt einen langsamen Applaus durch.",
        "parameters": {
            "type": "object",
            "properties": {
                "claps": {
                    "type": "number",
                    "description": "Die Anzahl der langsamen Klatschgeräusche."
                }
            },
            "required": ["claps"]
        }
    }
}

class SlowClap:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialisiert das Tool mit einer Queue zur Kommunikation mit dem LLM.

        Args:
            llm_queue: Eine Queue zum Senden von Tool-Ergebnissen an das Sprachmodell.
            tool_config: Konfigurationswörterbuch mit Tool-Einstellungen.
        """
        self.llm_queue = llm_queue
        tool_config = tool_config or {}
        self.audio_path = tool_config.get("slow_clap_audio_path", "data/slow-clap.mp3")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        """
        Führt den langsamen Applaus aus, indem eine Audiodatei mehrmals abgespielt wird.

        Args:
            tool_call_id: Eindeutiger Bezeichner für den Tool-Aufruf.
            call_args: Vom LLM übergebene Argumente für diesen Tool-Aufruf.
        """
        try:
            claps = int(call_args.get("claps", 1))
            claps = max(1, min(claps, 5))  # Begrenzen auf 1 bis 5
        except (ValueError, TypeError):
            claps = 1

        try:
            data, sample_rate = sf.read(self.audio_path)

            for _ in range(claps):
                sd.play(data, sample_rate)
                sd.wait()
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "success",
                    "type": "function_call_output",
                }
            )

        except FileNotFoundError:
            error_msg = f"Fehler: Audiodatei nicht gefunden unter {self.audio_path}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )

        except ValueError as ve:
            error_msg = f"Fehler: Ungültige Audiodatei - {ve}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )

        except sd.PortAudioError as pa_err:
            error_msg = f"Fehler: Audiogerät-Fehler - {pa_err}"
            logger.error(f"SlowClap: {error_msg}")
            self.llm_queue.put(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": error_msg,
                    "type": "function_call_output",
                }
            )
