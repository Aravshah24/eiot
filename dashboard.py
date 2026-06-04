from __future__ import annotations

import json
import time
from collections import deque

import cv2
import pandas as pd
import plotly.express as px
import serial
import streamlit as st

from camera_manager import create_camera_manager
from intrusion_detector import DEFAULT_POLYGON_POINTS
from esp32_alert import build_alert_command, send_serial_command
from vla_engine import decide_action

st.set_page_config(
    page_title="Smart Farmland Monitoring",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: #1c1f26;
        border: 1px solid #2c313c;
        border-radius: 14px;
        padding: 16px 18px;
    }
    div[data-testid="stMetricLabel"] {opacity: 0.8;}
    .alert-banner {
        border-radius: 14px;
        padding: 16px 20px;
        margin: 6px 0 18px 0;
        font-size: 1.1rem;
        font-weight: 600;
        border: 1px solid;
    }
    .alert-danger {background: #3a1416; border-color: #ff4b4b; color: #ff8585;}
    .alert-ok     {background: #11291a; border-color: #1fad5b; color: #4ade80;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌾 Smart Farmland Monitoring & Intrusion Detection System")

st.sidebar.header("System Settings")
camera_source_input = st.sidebar.text_input("Camera source", value="0")
serial_port = st.sidebar.text_input("ESP32 COM port", value="COM6")

st.sidebar.subheader("Pest Detection")
pest_confidence = st.sidebar.slider(
    "Pest confidence threshold", 0.10, 0.95, 0.60, 0.05,
    help="Higher = fewer false alarms. This model over-detects 'fruit fly' on backgrounds, so keep this high.",
)
pest_interval = st.sidebar.slider(
    "Pest check interval (s)", 1.0, 60.0, 2.0, 1.0,
    help="How often the pest model runs on the camera frame.",
)

camera_sidebar = st.sidebar.container()


def _parse_camera_source(raw_source: str) -> int | str:
    candidate = raw_source.strip()
    if candidate.isdigit():
        return int(candidate)
    return candidate


@st.cache_resource
def get_serial_connection(port: str):
    return serial.Serial(port, 115200, timeout=1)


@st.cache_resource
def get_camera_manager(source_value: int | str):
    return create_camera_manager(
        source=source_value,
        polygon_points=DEFAULT_POLYGON_POINTS,
        intrusion_interval=0.15,
        pest_interval=2.0,
        pest_confidence=0.6,
    )


def read_sensor_packet(connection) -> dict[str, float | int] | None:
    try:
        if connection.in_waiting <= 0:
            return None

        line = connection.readline().decode(errors="ignore").strip()
        if not line:
            return None

        data = json.loads(line)
        return {
            "temp": float(data.get("temp", 0.0)),
            "humidity": float(data.get("humidity", 0.0)),
            "pir": int(data.get("pir", 0)),
        }
    except Exception:
        return None


def sensor_snapshot_to_display(snapshot: dict[str, float | int] | None) -> dict[str, float | int]:
    if snapshot is not None:
        st.session_state["latest_sensor_snapshot"] = snapshot
        return snapshot

    return st.session_state.get(
        "latest_sensor_snapshot",
        {"temp": 0.0, "humidity": 0.0, "pir": 0},
    )


def frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


camera_manager = None
ser = None

try:
    camera_manager = get_camera_manager(_parse_camera_source(camera_source_input))
    # Live-tune the running detector from the sidebar sliders.
    camera_manager.pest_confidence = float(pest_confidence)
    camera_manager.pest_interval = float(pest_interval)
except Exception as exc:
    st.sidebar.error(f"Camera unavailable: {exc}")

try:
    ser = get_serial_connection(serial_port)
except Exception as exc:
    st.sidebar.error(f"Serial unavailable: {exc}")

temp_history = deque(maxlen=50)
humidity_history = deque(maxlen=50)

banner_placeholder = st.empty()
status_placeholder = st.empty()

last_sensor = st.session_state.get("latest_sensor_snapshot", {"temp": 0.0, "humidity": 0.0, "pir": 0})
ALERT_REFRESH_SECONDS = 5.0

while True:
    try:

        sensor_data = read_sensor_packet(ser) if ser is not None else None
        last_sensor = sensor_snapshot_to_display(sensor_data)

        temp = float(last_sensor["temp"])
        humidity = float(last_sensor["humidity"])
        pir = int(last_sensor["pir"])

        system_state = camera_manager.get_state() if camera_manager is not None else {
            "intrusion": False,
            "intrusion_object": None,
            "pest_detected": False,
            "pest_type": None,
            "last_pest_check": None,
        }
        decision = decide_action(system_state, last_sensor)

        intrusion_active = bool(system_state["intrusion"])
        intrusion_object = system_state.get("intrusion_object")
        pest_active = bool(system_state["pest_detected"])
        pest_type = system_state.get("pest_type")
        pir_active = bool(pir)

        # ---- Unified ESP32 buzzer/alert: ANY threat sounds the buzzer ----
        alert_active = intrusion_active or pir_active or pest_active
        active_reasons = set()
        if intrusion_active:
            active_reasons.add("intrusion")
        if pir_active:
            active_reasons.add("pir")
        if pest_active:
            active_reasons.add("pest")

        # Popup toasts only when a NEW reason appears (rising edge).
        previous_reasons = set(st.session_state.get("previous_alert_reasons", set()))
        new_reasons = active_reasons - previous_reasons
        if "intrusion" in new_reasons:
            st.toast(f"🚨 Intrusion: {intrusion_object or 'intruder'} in restricted area!", icon="🚨")
        if "pir" in new_reasons:
            st.toast("🚶 Physical breach detected by PIR sensor!", icon="🚶")
        if "pest" in new_reasons:
            st.toast(f"🐛 Pest detected: {pest_type or 'unknown'}!", icon="🐛")
        st.session_state["previous_alert_reasons"] = active_reasons

        # Resend on state change OR periodically while active (keeps buzzer alive).
        alert_signature = (alert_active, tuple(sorted(active_reasons)), pest_type, intrusion_object)
        last_signature = st.session_state.get("last_alert_signature")
        last_sent_at = float(st.session_state.get("last_alert_sent_at", 0.0))
        current_time = time.monotonic()
        should_send = (
            alert_signature != last_signature
            or (alert_active and current_time - last_sent_at >= ALERT_REFRESH_SECONDS)
        )

        if ser is not None and should_send:
            try:
                send_serial_command(
                    ser,
                    build_alert_command(
                        intrusion=intrusion_active,
                        pir=pir_active,
                        pest=pest_active,
                        pest_type=pest_type,
                        intrusion_object=intrusion_object,
                    ),
                )
                st.session_state["last_esp32_alert_status"] = (
                    f"Sent alert to ESP32: buzzer {'ON' if alert_active else 'OFF'}"
                    + (f" ({', '.join(sorted(active_reasons))})" if active_reasons else "")
                )
                st.session_state["last_alert_signature"] = alert_signature
                st.session_state["last_alert_sent_at"] = current_time
            except Exception as exc:
                st.sidebar.error(f"Failed to send ESP32 alert: {exc}")

        if sensor_data is not None:
            temp_history.append(temp)
            humidity_history.append(humidity)

        # ---- Top alert banner ----
        with banner_placeholder.container():
            if alert_active:
                labels = {
                    "intrusion": f"Virtual intrusion ({intrusion_object or 'intruder'})",
                    "pir": "Physical PIR breach",
                    "pest": f"Pest ({pest_type or 'unknown'})",
                }
                items = " · ".join(labels[r] for r in ("intrusion", "pir", "pest") if r in active_reasons)
                st.markdown(
                    f'<div class="alert-banner alert-danger">🔔 BUZZER ON — {items}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-banner alert-ok">✅ All clear — no active threats</div>',
                    unsafe_allow_html=True,
                )

        with status_placeholder.container():

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡 Temperature", f"{temp:.1f} °C")
            col2.metric("💧 Humidity", f"{humidity:.1f} %")
            col3.metric("🚶 PIR Status", "Motion" if pir else "Clear")
            col4.metric("🎯 Intrusion", "Detected" if intrusion_active else "Clear")

            st.divider()

            left, right = st.columns([3, 2])

            with left:
                st.subheader("📷 Live Camera (geofence · intrusion · pest)")
                if camera_manager is None:
                    st.warning("Camera manager is not running")
                else:
                    latest_frame = camera_manager.get_latest_frame()
                    if latest_frame is not None:
                        st.image(frame_to_rgb(latest_frame), use_container_width=True)
                    else:
                        st.info("Waiting for a captured frame...")

                if camera_manager is not None and getattr(camera_manager, "backend_window_error", None):
                    st.warning(f"Backend window error: {camera_manager.backend_window_error}")

            with right:
                st.subheader("Active Alerts")

                if intrusion_active:
                    st.error(f"🚨 Virtual intrusion: {intrusion_object}")
                if pir:
                    st.warning("🚶 Physical intrusion detected by PIR")
                if pest_active:
                    st.warning(f"🐛 Pest detected: {pest_type}")
                if temp > 35:
                    st.warning("🔥 High Temperature Alert")
                if humidity > 85:
                    st.warning("🦠 High Disease Risk")
                if not alert_active and temp <= 35 and humidity <= 85:
                    st.success("✅ No Active Alerts")

                st.info(f"🧠 Recommended Action: {decision['action']}")
                st.caption(st.session_state.get("last_esp32_alert_status", "ESP32 alert has not been sent yet"))
                st.metric("Last Pest Check", system_state["last_pest_check"] or "Pending")

            st.divider()

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Temperature Trend")
                temp_df = pd.DataFrame({
                    "Reading": list(range(len(temp_history))),
                    "Temperature": list(temp_history),
                })
                fig1 = px.line(temp_df, x="Reading", y="Temperature")
                st.plotly_chart(fig1, use_container_width=True, key="temperature_trend_chart")

            with chart_col2:
                st.subheader("Humidity Trend")
                hum_df = pd.DataFrame({
                    "Reading": list(range(len(humidity_history))),
                    "Humidity": list(humidity_history),
                })
                fig2 = px.line(hum_df, x="Reading", y="Humidity")
                st.plotly_chart(fig2, use_container_width=True, key="humidity_trend_chart")

            st.divider()

            st.subheader("🧠 VLA Decision Engine")
            decisions = []
            if intrusion_active:
                decisions.append(f"Intrusion detected → Alert farmer + sound buzzer ({intrusion_object})")
            if pir_active:
                decisions.append("Physical breach (PIR) → Sound buzzer")
            if pest_active:
                decisions.append(f"Pest detected → Recommend pest control for {pest_type} + sound buzzer")
            if temp > 35:
                decisions.append("Temperature high → Monitor crops")
            if humidity > 85:
                decisions.append("Humidity high → Disease risk")
            if not decisions:
                decisions.append("Normal conditions")
            for d in decisions:
                st.write("•", d)

            if alert_active:
                st.warning("ESP32 alert active: buzzer and light flashing")
            else:
                st.caption("ESP32 alert is idle")

        with camera_sidebar:
            st.subheader("📷 Camera Window")
            if camera_manager is None:
                st.warning("Camera manager unavailable")
            else:
                sidebar_frame = camera_manager.get_latest_frame()
                if sidebar_frame is not None:
                    st.image(frame_to_rgb(sidebar_frame), caption="Live camera window", use_container_width=True)
                else:
                    st.info("Waiting for camera window...")

    except Exception as e:
        st.error(str(e))

    time.sleep(1)
