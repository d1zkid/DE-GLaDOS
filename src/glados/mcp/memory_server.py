"""
Langzeitgedächtnis-MCP-Server für GLaDOS.

Stellt Werkzeuge zum Speichern und Abrufen von Fakten und Zusammenfassungen bereit.
Verwendet das LLM für semantisches Such-Ranking (LLM-first-Prinzip).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("memory")

# Speicherpfade
MEMORY_DIR = Path(os.path.expanduser("~/.glados/memory"))
FACTS_FILE = MEMORY_DIR / "facts.jsonl"
SUMMARIES_FILE = MEMORY_DIR / "summaries.jsonl"


@dataclass
class Fact:
    """Ein gespeicherter Fakt mit Metadaten."""

    content: str
    source: str
    importance: float
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"fact_{int(time.time() * 1000)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        return cls(**data)


@dataclass
class Summary:
    """Eine gespeicherte Gesprächszusammenfassung."""

    content: str
    period: str  # "daily", "weekly", "session"
    start_time: str  # ISO-Format
    end_time: str  # ISO-Format
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"summary_{int(time.time() * 1000)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Summary":
        return cls(**data)


def _ensure_storage() -> None:
    """Stellt sicher, dass das Speicherverzeichnis existiert."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _load_facts() -> list[Fact]:
    """Lädt alle gespeicherten Fakten."""
    if not FACTS_FILE.exists():
        return []
    facts = []
    try:
        with FACTS_FILE.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    facts.append(Fact.from_dict(json.loads(line)))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Fakten konnten nicht geladen werden: {e}")
    return facts


def _save_fact(fact: Fact) -> None:
    """Hängt einen Fakt an den Speicher an."""
    _ensure_storage()
    with FACTS_FILE.open("a") as f:
        f.write(json.dumps(fact.to_dict()) + "\n")


def _load_summaries() -> list[Summary]:
    """Lädt alle gespeicherten Zusammenfassungen."""
    if not SUMMARIES_FILE.exists():
        return []
    summaries = []
    try:
        with SUMMARIES_FILE.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    summaries.append(Summary.from_dict(json.loads(line)))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Zusammenfassungen konnten nicht geladen werden: {e}")
    return summaries


def _save_summary(summary: Summary) -> None:
    """Hängt eine Zusammenfassung an den Speicher an."""
    _ensure_storage()
    with SUMMARIES_FILE.open("a") as f:
        f.write(json.dumps(summary.to_dict()) + "\n")


@mcp.tool()
def store_fact(fact: str, source: str = "user", importance: float = 0.5) -> str:
    """
    Speichert einen Fakt im Langzeitgedächtnis.

    Args:
        fact: Der zu speichernde Fakt (z. B. „Der Name des Nutzers ist David")
        source: Herkunft des Fakts („user", „conversation", „system")
        importance: Wichtigkeit des Fakts (0.0 bis 1.0)

    Returns:
        Bestätigungsmeldung mit der Fakt-ID
    """
    importance = max(0.0, min(1.0, importance))
    new_fact = Fact(content=fact, source=source, importance=importance)
    _save_fact(new_fact)
    return json.dumps({
        "status": "gespeichert",
        "id": new_fact.id,
        "fact": fact,
    })


@mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """
    Durchsucht das Langzeitgedächtnis nach relevanten Fakten.

    Verwendet einfaches Schlüsselwort-Matching. Für semantische Suche
    sollte der Hauptagent die Ergebnisse im Kontext interpretieren.

    Args:
        query: Wonach gesucht werden soll
        limit: Maximale Anzahl der Ergebnisse (Standard: 5)

    Returns:
        JSON-Array mit passenden Fakten, sortiert nach Relevanz
    """
    facts = _load_facts()
    if not facts:
        return json.dumps({"facts": [], "message": "Noch keine Fakten gespeichert"})

    # Einfaches schlüsselwortbasiertes Scoring (Semantik übernimmt der Hauptagent)
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored = []
    for fact in facts:
        content_lower = fact.content.lower()
        # Score: Wortüberschneidung + Wichtigkeitsbonus + Aktualität
        word_score = sum(1 for w in query_words if w in content_lower)
        importance_boost = fact.importance * 0.5
        recency_boost = min(0.3, (time.time() - fact.created_at) / (86400 * 30))  # Abfall über 30 Tage
        total_score = word_score + importance_boost - recency_boost

        if word_score > 0 or query_lower in content_lower:
            scored.append((total_score, fact))

    # Absteigend nach Score sortieren
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, fact in scored[:limit]:
        results.append({
            "id": fact.id,
            "content": fact.content,
            "source": fact.source,
            "importance": fact.importance,
            "created_at": datetime.fromtimestamp(fact.created_at).isoformat(),
        })

    return json.dumps({"facts": results, "total_stored": len(facts)})


@mcp.tool()
def list_facts(limit: int = 20, min_importance: float = 0.0) -> str:
    """
    Listet gespeicherte Fakten auf, optional nach Wichtigkeit gefiltert.

    Args:
        limit: Maximale Anzahl zurückzugebender Fakten
        min_importance: Mindestwichtigkeitsschwelle (0.0 bis 1.0)

    Returns:
        JSON-Array mit Fakten, sortiert nach Wichtigkeit und Aktualität
    """
    facts = _load_facts()
    filtered = [f for f in facts if f.importance >= min_importance]

    # Sortierung: Wichtigkeit (absteigend), dann Aktualität (absteigend)
    filtered.sort(key=lambda f: (f.importance, f.created_at), reverse=True)

    results = []
    for fact in filtered[:limit]:
        results.append({
            "id": fact.id,
            "content": fact.content,
            "source": fact.source,
            "importance": fact.importance,
            "created_at": datetime.fromtimestamp(fact.created_at).isoformat(),
        })

    return json.dumps({"facts": results, "total_stored": len(facts)})


@mcp.tool()
def store_summary(summary: str, period: str, start_time: str, end_time: str) -> str:
    """
    Speichert eine Gesprächszusammenfassung.

    Args:
        summary: Der Zusammenfassungstext
        period: Zusammenfassungszeitraum („session", „daily", „weekly")
        start_time: Beginn des zusammengefassten Zeitraums (ISO-Format)
        end_time: Ende des zusammengefassten Zeitraums (ISO-Format)

    Returns:
        Bestätigungsmeldung mit der Zusammenfassungs-ID
    """
    if period not in ("session", "daily", "weekly"):
        return json.dumps({"error": "period muss 'session', 'daily' oder 'weekly' sein"})

    new_summary = Summary(
        content=summary,
        period=period,
        start_time=start_time,
        end_time=end_time,
    )
    _save_summary(new_summary)
    return json.dumps({
        "status": "gespeichert",
        "id": new_summary.id,
        "period": period,
    })


@mcp.tool()
def get_summaries(period: str = "all", limit: int = 5) -> str:
    """
    Ruft gespeicherte Gesprächszusammenfassungen ab.

    Args:
        period: Filterung nach Zeitraum („session", „daily", „weekly" oder „all")
        limit: Maximale Anzahl zurückzugebender Zusammenfassungen

    Returns:
        JSON-Array mit Zusammenfassungen, neueste zuerst
    """
    summaries = _load_summaries()

    if period != "all":
        summaries = [s for s in summaries if s.period == period]

    # Sortierung nach created_at absteigend (neueste zuerst)
    summaries.sort(key=lambda s: s.created_at, reverse=True)

    results = []
    for summary in summaries[:limit]:
        results.append({
            "id": summary.id,
            "content": summary.content,
            "period": summary.period,
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "created_at": datetime.fromtimestamp(summary.created_at).isoformat(),
        })

    return json.dumps({"summaries": results, "total_stored": len(_load_summaries())})


@mcp.tool()
def memory_stats() -> str:
    """
    Gibt Statistiken über gespeicherte Erinnerungen zurück.

    Returns:
        JSON-Objekt mit Gedächtnisstatistiken
    """
    facts = _load_facts()
    summaries = _load_summaries()

    source_counts: dict[str, int] = {}
    for fact in facts:
        source_counts[fact.source] = source_counts.get(fact.source, 0) + 1

    period_counts: dict[str, int] = {}
    for summary in summaries:
        period_counts[summary.period] = period_counts.get(summary.period, 0) + 1

    avg_importance = sum(f.importance for f in facts) / len(facts) if facts else 0

    return json.dumps({
        "total_facts": len(facts),
        "total_summaries": len(summaries),
        "facts_by_source": source_counts,
        "summaries_by_period": period_counts,
        "average_importance": round(avg_importance, 2),
    })


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
