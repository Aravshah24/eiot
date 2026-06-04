from __future__ import annotations

from typing import Any, Mapping


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def decide_action(system_state: Mapping[str, Any], sensor_data: Mapping[str, Any]) -> dict[str, str]:
    intrusion = bool(system_state.get("intrusion"))
    pest_detected = bool(system_state.get("pest_detected"))
    humidity = _to_float(sensor_data.get("humidity"))
    temperature = _to_float(sensor_data.get("temp", sensor_data.get("temperature")))

    if intrusion:
        return {
            "action": "Alert Farmer",
            "alert_type": "intrusion",
            "esp32_alert": {
                "active": False,
                "buzzer": 0,
                "light": 0,
                "flash": 0,
            },
        }

    if pest_detected:
        pest_type = system_state.get("pest_type") or "Pest Control"
        return {
            "action": f"Recommend Pest Control for {pest_type}",
            "alert_type": "pest",
            "esp32_alert": {
                "active": True,
                "buzzer": 1,
                "light": 1,
                "flash": 1,
                "pest_type": pest_type,
            },
        }

    if humidity > 85:
        return {
            "action": "Fungal Disease Warning",
            "alert_type": "humidity",
            "esp32_alert": {
                "active": False,
                "buzzer": 0,
                "light": 0,
                "flash": 0,
            },
        }

    if temperature > 35:
        return {
            "action": "Heat Stress Warning",
            "alert_type": "temperature",
            "esp32_alert": {
                "active": False,
                "buzzer": 0,
                "light": 0,
                "flash": 0,
            },
        }

    return {
        "action": "Normal Monitoring",
        "alert_type": "normal",
        "esp32_alert": {
            "active": False,
            "buzzer": 0,
            "light": 0,
            "flash": 0,
        },
    }