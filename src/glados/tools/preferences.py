"""
Einstellungs-Tools zur Verwaltung von Benutzereinstellungen.

Diese Tools ermöglichen es dem Hauptagenten, Benutzereinstellungen abzurufen
und zu setzen. Subagenten können diese lesen, um ihr Verhalten anzupassen.
"""

import json
import queue
from typing import Any


get_preferences_definition = {
    "type": "function",
    "function": {
        "name": "get_preferences",
        "description": "Aktuelle Benutzereinstellungen abrufen. Gibt alle gespeicherten Einstellungen als JSON zurück.",
        "parameters": {"type": "object", "properties": {}},
    },
}


set_preference_definition = {
    "type": "function",
    "function": {
        "name": "set_preference",
        "description": (
            "Eine Benutzereinstellung festlegen. Verwenden, um Vorlieben und Abneigungen des Benutzers zu speichern. "
            "Beispiele: news_topics=['KI', 'Wissenschaft'], news_exclude=['Krypto'], weather_units='celsius'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Einstellungsschlüssel (z. B. 'news_topics', 'weather_units')",
                },
                "value": {
                    "description": "Einstellungswert (Zeichenkette, Zahl, Boolescher Wert oder Array)",
                },
            },
            "required": ["key", "value"],
        },
    },
}


class GetPreferences:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.preferences_store = (tool_config or {}).get("preferences_store")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if not self.preferences_store:
            result = "Fehler: Einstellungsspeicher nicht konfiguriert"
        else:
            prefs = self.preferences_store.all()
            result = json.dumps(prefs, indent=2) if prefs else "{}"
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
                "type": "function_call_output",
            }
        )


class SetPreference:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.preferences_store = (tool_config or {}).get("preferences_store")

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        if not self.preferences_store:
            result = "Fehler: Einstellungsspeicher nicht konfiguriert"
        else:
            key = call_args.get("key", "")
            value = call_args.get("value")
            if not key:
                result = "Fehler: Schlüssel ist erforderlich"
            else:
                self.preferences_store.set(key, value)
                result = f"Gesetzt: {key} = {json.dumps(value)}"
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
                "type": "function_call_output",
            }
        )
