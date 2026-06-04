from __future__ import annotations

import json
from typing import Any, Mapping


def build_alert_command(
    *,
    intrusion: bool = False,
    pir: bool = False,
    pest: bool = False,
    pest_type: str | None = None,
    intrusion_object: str | None = None,
) -> dict[str, Any]:
    """Single command the ESP32 reads to drive the buzzer/light.

    The buzzer fires for ANY active threat: virtual intrusion, PIR (physical
    geofence) breach, or pest detection. A simple firmware only needs to read
    the ``buzzer`` field; ``reasons`` is extra context for richer firmware.
    """
    active = bool(intrusion or pir or pest)

    reasons: list[str] = []
    if intrusion:
        reasons.append("intrusion")
    if pir:
        reasons.append("pir")
    if pest:
        reasons.append("pest")

    payload: dict[str, Any] = {
        "type": "farm_alert",
        "active": active,
        "buzzer": 1 if active else 0,
        "light": 1 if active else 0,
        "flash": 1 if active else 0,
        "reasons": reasons,
    }

    if pest_type:
        payload["pest_type"] = pest_type
    if intrusion_object:
        payload["intrusion_object"] = intrusion_object

    return payload


def send_serial_command(connection: Any, payload: Mapping[str, Any]) -> None:
    message = json.dumps(dict(payload), separators=(",", ":")) + "\n"
    connection.write(message.encode("utf-8"))
    connection.flush()
