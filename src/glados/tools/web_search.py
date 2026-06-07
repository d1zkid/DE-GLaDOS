import os
import queue
from pathlib import Path
from typing import Any

import serpapi
from dotenv import load_dotenv
from loguru import logger

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")

tool_definition = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Durchsucht das Web nach aktuellen Informationen über SerpAPI. "
            "Verwende dieses Tool IMMER wenn nach Folgendem gefragt wird: aktuelle Ereignisse, "
            "Erscheinungsdaten, Preise, Nachrichten, neue Entwicklungen, Sportergebnisse, "
            "Veröffentlichungsdaten von Filmen/Serien/Spielen oder JEDES Thema, bei dem deine "
            "Trainingsdaten veraltet oder unvollständig sein könnten. "
            "Im Zweifel ob eine Information aktuell ist, dieses Tool verwenden. "
            "Antworte NICHT aus dem Gedächtnis, wenn das Thema sich seit deinem Training geändert haben könnte."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Suchanfrage, die an SerpAPI gesendet werden soll.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Optionale maximale Anzahl der zu berücksichtigenden Suchergebnisse (Standard: 5, maximal 10).",
                },
            },
            "required": ["query"],
        },
    },
}


class WebSearch:
    def __init__(
        self,
        llm_queue: queue.Queue[dict[str, Any]],
        tool_config: dict[str, Any] | None = None,
    ) -> None:
        self.llm_queue = llm_queue
        self.tool_config = tool_config or {}
        self.api_key = os.getenv("SERPAPI_KEY") or self.tool_config.get("serpapi_key", "")
        self.completion_url = self.tool_config.get("completion_url")
        self.llm_model = self.tool_config.get("llm_model")
        self.llm_headers = self.tool_config.get("llm_headers") or {}

    def run(self, tool_call_id: str, call_args: dict[str, Any]) -> None:
        query = call_args.get("query", "").strip()
        max_results = call_args.get("max_results", 5)

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 5
        max_results = max(1, min(max_results, 10))

        if not query:
            self._send_response(tool_call_id, "Fehler: Die Suchanfrage (query) darf nicht leer sein.")
            return

        if not self.api_key:
            self._send_response(tool_call_id, "Fehler: Kein SerpAPI-Schlüssel konfiguriert (serpapi_key).")
            return

        logger.info(f"WebSearch: Suche nach '{query}' mit SerpAPI...")

        try:
            client = serpapi.Client(api_key=self.api_key)
            results = client.search({"q": query})
        except Exception as e:
            logger.error(f"WebSearch: Fehler bei der SerpAPI-Anfrage: {e}")
            self._send_response(tool_call_id, f"Fehler bei der Websuche: {e}")
            return

        content = self._extract_content(results, query, max_results)
        logger.info("WebSearch: Ergebnisse erfolgreich extrahiert.")
        self._send_response(tool_call_id, content)

    def _extract_content(self, results: dict, query: str, max_results: int) -> str:
        """Extracts the best available content from SerpAPI results."""
        parts = []

        # 1. AI Overview (richest source — structured text blocks)
        ai_overview = results.get("ai_overview")
        if ai_overview and "text_blocks" in ai_overview:
            logger.info("WebSearch: AI Overview gefunden.")
            overview_lines = []
            for block in ai_overview["text_blocks"]:
                block_type = block.get("type")
                if block_type == "paragraph":
                    overview_lines.append(block.get("snippet", ""))
                elif block_type == "heading":
                    overview_lines.append(f"\n{block.get('snippet', '')}")
                elif block_type == "list":
                    for item in block.get("list", []):
                        overview_lines.append(f"  - {item.get('snippet', '')}")
            if overview_lines:
                parts.append("AI Übersicht:\n" + "\n".join(overview_lines))

        # 2. Answer box (e.g. for direct factual queries)
        answer_box = results.get("answer_box")
        if answer_box:
            answer = (
                answer_box.get("answer")
                or answer_box.get("snippet")
                or answer_box.get("result")
            )
            if answer:
                parts.append(f"Direkte Antwort: {answer}")

        # 3. Knowledge graph (e.g. for people, places, things)
        knowledge_graph = results.get("knowledge_graph")
        if knowledge_graph:
            kg_parts = []
            if knowledge_graph.get("title"):
                kg_parts.append(knowledge_graph["title"])
            if knowledge_graph.get("description"):
                kg_parts.append(knowledge_graph["description"])
            if kg_parts:
                parts.append("Wissensgraph: " + " — ".join(kg_parts))

        # 4. Organic results (standard search results)
        organic = results.get("organic_results", [])
        if organic:
            snippets = []
            for r in organic[:max_results]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                if title or snippet:
                    snippets.append(f"- {title}: {snippet}")
            if snippets:
                parts.append("Suchergebnisse:\n" + "\n".join(snippets))

        if parts:
            return "\n\n".join(parts)

        return f"Keine Ergebnisse für '{query}' gefunden."

    def _send_response(self, tool_call_id: str, content: str) -> None:
        self.llm_queue.put(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
                "type": "function_call_output",
            }
        )