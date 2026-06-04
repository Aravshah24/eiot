from __future__ import annotations

import json
from typing import Any, Mapping


def build_pest_alert_command(pest_type: str | None, active: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "pest_alert",
        "active": bool(active),
        "buzzer": 1 if active else 0,
        "light": 1 if active else 0,
        "flash": 1 if active else 0,
    }

    if pest_type:
        payload["pest_type"] = pest_type

    return payload


def send_serial_command(connection: Any, payload: Mapping[str, Any]) -> None:
    message = json.dumps(dict(payload), separators=(",", ":")) + "\n"
    connection.write(message.encode("utf-8"))
    connection.flush()
