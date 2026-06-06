"""
GLaDOS MCP Server – Systeminfo (Deutsch)
Liefert CPU, RAM, GPU, Festplatte und Temperaturen auf Deutsch.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("system_info")


# ─────────────────────────────────────────────
# Hilfsfunktionen
# ─────────────────────────────────────────────

def _bytes_to_gb(b: int | float) -> float:
    return round(b / (1024 ** 3), 2)


def _read_meminfo() -> dict[str, int] | None:
    p = Path("/proc/meminfo")
    if not p.exists():
        return None
    data: dict[str, int] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            # Values in /proc/meminfo are in kB
            data[key.strip()] = int(parts[0]) * 1024
        except ValueError:
            continue
    return data


def _read_temps_sysfs() -> list[dict[str, Any]]:
    readings: list[dict[str, Any]] = []

    thermal_root = Path("/sys/class/thermal")
    if thermal_root.exists():
        for zone in sorted(thermal_root.glob("thermal_zone*")):
            temp_path = zone / "temp"
            if not temp_path.exists():
                continue
            try:
                milli_c = int(temp_path.read_text(encoding="utf-8").strip())
            except ValueError:
                continue
            label = (
                (zone / "type").read_text(encoding="utf-8").strip()
                if (zone / "type").exists()
                else zone.name
            )
            readings.append({"sensor": label, "celsius": round(milli_c / 1000.0, 1)})

    hwmon_root = Path("/sys/class/hwmon")
    if hwmon_root.exists():
        for hwmon in sorted(hwmon_root.glob("hwmon*")):
            name = (
                (hwmon / "name").read_text(encoding="utf-8").strip()
                if (hwmon / "name").exists()
                else hwmon.name
            )
            for temp_input in sorted(hwmon.glob("temp*_input")):
                try:
                    milli_c = int(temp_input.read_text(encoding="utf-8").strip())
                except ValueError:
                    continue
                sensor_id = temp_input.stem.replace("_input", "")
                label_path = hwmon / f"{sensor_id}_label"
                label = (
                    label_path.read_text(encoding="utf-8").strip()
                    if label_path.exists()
                    else sensor_id
                )
                readings.append({
                    "sensor": f"{name}:{label}",
                    "celsius": round(milli_c / 1000.0, 1),
                })

    return readings


def _get_gpu_info() -> dict[str, Any]:
    """Versucht GPU-Infos via nvidia-smi oder ROCm zu lesen."""
    # NVIDIA via pynvml (bevorzugt)
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "name": name,
                "auslastung_prozent": util.gpu,
                "vram_genutzt_gb": _bytes_to_gb(mem.used),
                "vram_gesamt_gb": _bytes_to_gb(mem.total),
                "vram_auslastung_prozent": round(mem.used / mem.total * 100, 1),
            })
        pynvml.nvmlShutdown()
        return {"gpus": gpus}
    except Exception:
        pass

    # Fallback: nvidia-smi subprocess
    try:
        import subprocess
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    used_mb = int(parts[2])
                    total_mb = int(parts[3])
                    gpus.append({
                        "name": parts[0],
                        "auslastung_prozent": int(parts[1]),
                        "vram_genutzt_gb": round(used_mb / 1024, 2),
                        "vram_gesamt_gb": round(total_mb / 1024, 2),
                        "vram_auslastung_prozent": round(used_mb / total_mb * 100, 1),
                    })
            return {"gpus": gpus}
    except Exception:
        pass

    return {"fehler": "Keine GPU-Informationen verfügbar (kein nvidia-smi oder pynvml)"}


# ─────────────────────────────────────────────
# MCP Tools
# ─────────────────────────────────────────────

@mcp.tool()
def cpu_auslastung() -> str:
    """Gibt die CPU-Auslastung in Prozent zurück (1-Minuten-Lastdurchschnitt)."""
    try:
        import psutil  # type: ignore
        percent = psutil.cpu_percent(interval=0.5)
        count_logisch = psutil.cpu_count(logical=True)
        count_physisch = psutil.cpu_count(logical=False)
        result = {
            "auslastung_prozent": percent,
            "kerne_logisch": count_logisch,
            "kerne_physisch": count_physisch,
            "bewertung": (
                "normal" if percent < 70
                else "erhöht" if percent < 90
                else "KRITISCH – System stark ausgelastet"
            ),
        }
    except ImportError:
        try:
            one, five, fifteen = os.getloadavg()
            result = {
                "last_1min": round(one, 2),
                "last_5min": round(five, 2),
                "last_15min": round(fifteen, 2),
                "hinweis": "psutil nicht verfügbar – Lastdurchschnitt statt Prozent",
            }
        except (AttributeError, OSError):
            return json.dumps({"fehler": "CPU-Auslastung nicht verfügbar"})

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def ram_auslastung() -> str:
    """Gibt RAM-Nutzung zurück: Prozent, genutzter und maximaler Speicher in GB."""
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        result = {
            "auslastung_prozent": mem.percent,
            "genutzt_gb": _bytes_to_gb(mem.used),
            "verfuegbar_gb": _bytes_to_gb(mem.available),
            "gesamt_gb": _bytes_to_gb(mem.total),
            "bewertung": (
                "normal" if mem.percent < 75
                else "erhöht" if mem.percent < 90
                else "KRITISCH – Arbeitsspeicher fast voll"
            ),
        }
    except ImportError:
        meminfo = _read_meminfo()
        if not meminfo:
            return json.dumps({"fehler": "Arbeitsspeicher-Info nicht verfügbar"})
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        used = max(total - available, 0)
        percent = round(used / total * 100.0, 1) if total else 0.0
        result = {
            "auslastung_prozent": percent,
            "genutzt_gb": _bytes_to_gb(used),
            "verfuegbar_gb": _bytes_to_gb(available),
            "gesamt_gb": _bytes_to_gb(total),
            "bewertung": (
                "normal" if percent < 75
                else "erhöht" if percent < 90
                else "KRITISCH – Arbeitsspeicher fast voll"
            ),
        }

    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def festplatten_auslastung() -> str:
    """Gibt Festplattennutzung zurück: freier Speicher in GB und Auslastung in Prozent."""
    try:
        import psutil  # type: ignore
        partitions = psutil.disk_partitions(all=False)
        laufwerke = []
        for part in partitions:
            try:
                usage = psutil.disk_usage(part.mountpoint)
                laufwerke.append({
                    "pfad": part.mountpoint,
                    "dateisystem": part.fstype,
                    "gesamt_gb": _bytes_to_gb(usage.total),
                    "genutzt_gb": _bytes_to_gb(usage.used),
                    "frei_gb": _bytes_to_gb(usage.free),
                    "auslastung_prozent": usage.percent,
                    "bewertung": (
                        "normal" if usage.percent < 80
                        else "erhöht" if usage.percent < 90
                        else "KRITISCH – Festplatte fast voll"
                    ),
                })
            except PermissionError:
                continue
        return json.dumps({"laufwerke": laufwerke}, ensure_ascii=False)
    except ImportError:
        # Fallback via /proc/mounts + statvfs
        laufwerke = []
        try:
            for line in Path("/proc/mounts").read_text().splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                mountpoint = parts[1]
                try:
                    st = os.statvfs(mountpoint)
                    total = st.f_blocks * st.f_frsize
                    free = st.f_bfree * st.f_frsize
                    used = total - free
                    percent = round(used / total * 100, 1) if total else 0
                    laufwerke.append({
                        "pfad": mountpoint,
                        "gesamt_gb": _bytes_to_gb(total),
                        "genutzt_gb": _bytes_to_gb(used),
                        "frei_gb": _bytes_to_gb(free),
                        "auslastung_prozent": percent,
                    })
                except (OSError, ZeroDivisionError):
                    continue
        except Exception:
            return json.dumps({"fehler": "Festplatteninformationen nicht verfügbar"})
        return json.dumps({"laufwerke": laufwerke}, ensure_ascii=False)


@mcp.tool()
def temperaturen() -> str:
    """Gibt Temperatursensoren in Celsius zurück (CPU, Mainboard, etc.)."""
    readings = _read_temps_sysfs()

    # psutil als Ergänzung
    try:
        import psutil  # type: ignore
        if hasattr(psutil, "sensors_temperatures"):
            for name, entries in (psutil.sensors_temperatures() or {}).items():
                for e in entries:
                    label = f"{name}:{e.label}" if e.label else name
                    # Deduplizierung
                    if not any(r["sensor"] == label for r in readings):
                        readings.append({
                            "sensor": label,
                            "celsius": round(e.current, 1),
                        })
    except ImportError:
        pass

    if not readings:
        return json.dumps({"fehler": "Keine Temperatursensoren verfügbar"}, ensure_ascii=False)

    # Bewertung hinzufügen
    for r in readings:
        c = r["celsius"]
        r["bewertung"] = (
            "normal" if c < 70
            else "warm" if c < 85
            else "KRITISCH – Überhitzungsgefahr"
        )

    return json.dumps({"sensoren": readings}, ensure_ascii=False)


@mcp.tool()
def gpu_auslastung() -> str:
    """Gibt GPU-Auslastung in Prozent und VRAM-Nutzung in GB zurück."""
    return json.dumps(_get_gpu_info(), ensure_ascii=False)


@mcp.tool()
def system_uebersicht() -> str:
    """
    Gibt eine vollständige Systemübersicht zurück: CPU, RAM, GPU, Festplatte und Temperaturen.
    Ideal für eine schnelle Diagnose. Enthält Bewertungen ob etwas auffällig ist.
    Nutze dieses Tool wenn der Nutzer fragt ob das System langsam ist oder etwas nicht stimmt.
    """
    uebersicht: dict[str, Any] = {
        "cpu": json.loads(cpu_auslastung()),
        "ram": json.loads(ram_auslastung()),
        "festplatte": json.loads(festplatten_auslastung()),
        "temperaturen": json.loads(temperaturen()),
        "gpu": json.loads(gpu_auslastung()),
    }

    # Zusammenfassung der Warnungen
    warnungen = []

    cpu = uebersicht.get("cpu", {})
    if cpu.get("auslastung_prozent", 0) >= 90:
        warnungen.append("CPU-Auslastung kritisch hoch")
    elif cpu.get("auslastung_prozent", 0) >= 70:
        warnungen.append("CPU-Auslastung erhöht")

    ram = uebersicht.get("ram", {})
    if ram.get("auslastung_prozent", 0) >= 90:
        warnungen.append("Arbeitsspeicher fast voll")
    elif ram.get("auslastung_prozent", 0) >= 75:
        warnungen.append("Arbeitsspeicher-Auslastung erhöht")

    disk = uebersicht.get("festplatte", {})
    for lw in disk.get("laufwerke", []):
        if lw.get("auslastung_prozent", 0) >= 90:
            warnungen.append(f"Festplatte {lw['pfad']} fast voll ({lw['frei_gb']} GB frei)")
        elif lw.get("auslastung_prozent", 0) >= 80:
            warnungen.append(f"Festplatte {lw['pfad']} Auslastung erhöht")

    temps = uebersicht.get("temperaturen", {})
    for sensor in temps.get("sensoren", []):
        if sensor.get("celsius", 0) >= 85:
            warnungen.append(f"Sensor '{sensor['sensor']}' überhitzt: {sensor['celsius']}°C")

    gpu_data = uebersicht.get("gpu", {})
    for gpu in gpu_data.get("gpus", []):
        if gpu.get("auslastung_prozent", 0) >= 90:
            warnungen.append(f"GPU '{gpu['name']}' stark ausgelastet")
        if gpu.get("vram_auslastung_prozent", 0) >= 90:
            warnungen.append(f"GPU '{gpu['name']}' VRAM fast voll")

    uebersicht["warnungen"] = warnungen if warnungen else ["Alles im normalen Bereich"]
    uebersicht["system_status"] = "WARNUNG" if warnungen else "OK"

    return json.dumps(uebersicht, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()