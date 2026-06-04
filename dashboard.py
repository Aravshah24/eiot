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
from esp32_alert import build_pest_alert_command, send_serial_command
from vla_engine import decide_action

st.set_page_config(
    page_title="Smart Farmland Monitoring",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Smart Farmland Monitoring & Intrusion Detection System")

st.sidebar.header("System Settings")
camera_source_input = st.sidebar.text_input("Camera source", value="0")
serial_port = st.sidebar.text_input("ESP32 COM port", value="COM6")

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
        pest_interval=60.0,
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
except Exception as exc:
    st.sidebar.error(f"Camera unavailable: {exc}")

try:
    ser = get_serial_connection(serial_port)
except Exception as exc:
    st.sidebar.error(f"Serial unavailable: {exc}")

temp_history = deque(maxlen=50)
humidity_history = deque(maxlen=50)

camera_placeholder = st.empty()
status_placeholder = st.empty()
chart_placeholder = st.empty()

last_sensor = st.session_state.get("latest_sensor_snapshot", {"temp": 0.0, "humidity": 0.0, "pir": 0})
last_pest_alert_state = st.session_state.get("last_pest_alert_state", False)
last_pest_alert_type = st.session_state.get("last_pest_alert_type")
last_pest_alert_sent_at = float(st.session_state.get("last_pest_alert_sent_at", 0.0))
PEST_ALERT_REFRESH_SECONDS = 5.0

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

        pest_alert_active = bool(system_state["pest_detected"])
        pest_alert_type = system_state.get("pest_type")
        current_time = time.monotonic()
        should_send_pest_alert = (
            pest_alert_active != last_pest_alert_state
            or pest_alert_type != last_pest_alert_type
            or (
                pest_alert_active
                and current_time - last_pest_alert_sent_at >= PEST_ALERT_REFRESH_SECONDS
            )
        )

        if ser is not None and should_send_pest_alert:
            try:
                send_serial_command(
                    ser,
                    build_pest_alert_command(pest_alert_type, pest_alert_active)
                )
                st.session_state["last_esp32_alert_status"] = (
                    f"Sent pest alert to ESP32: {'ON' if pest_alert_active else 'OFF'}"
                )
                last_pest_alert_state = pest_alert_active
                last_pest_alert_type = pest_alert_type
                last_pest_alert_sent_at = current_time
                st.session_state["last_pest_alert_state"] = last_pest_alert_state
                st.session_state["last_pest_alert_type"] = last_pest_alert_type
                st.session_state["last_pest_alert_sent_at"] = last_pest_alert_sent_at
            except Exception as exc:
                st.sidebar.error(f"Failed to send ESP32 alert: {exc}")

        if sensor_data is not None:
            temp_history.append(temp)
            humidity_history.append(humidity)

        with status_placeholder.container():

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "🌡 Temperature",
                f"{temp:.1f} °C"
            )

            col2.metric(
                "💧 Humidity",
                f"{humidity:.1f} %"
            )

            col3.metric(
                "🚨 PIR Status",
                "Motion" if pir else "Clear"
            )

            col4.metric(
                "🎯 Intrusion Status",
                "Intrusion" if system_state["intrusion"] else "Clear"
            )

            st.divider()

            st.subheader("Active Alerts")

            if system_state["intrusion"]:
                st.error(
                    f"🚨 Virtual intrusion detected: {system_state['intrusion_object']}"
                )

            if pir:
                st.warning("🚨 Physical intrusion detected by PIR")

            if temp > 35:
                st.warning("🔥 High Temperature Alert")

            if humidity > 85:
                st.warning("🦠 High Disease Risk")

            if system_state["pest_detected"]:
                st.warning(
                    f"🐛 Pest detected: {system_state['pest_type']}"
                )

            if not pir and temp <= 35 and humidity <= 85 and not system_state["intrusion"] and not system_state["pest_detected"]:
                st.success("✅ No Active Alerts")

            st.info(f"🧠 Recommended Action: {decision['action']}")
            st.caption(st.session_state.get("last_esp32_alert_status", "ESP32 alert has not been sent yet"))

            st.divider()

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:

                st.subheader("Temperature Trend")

                temp_df = pd.DataFrame({
                    "Reading": list(range(len(temp_history))),
                    "Temperature": list(temp_history)
                })

                fig1 = px.line(
                    temp_df,
                    x="Reading",
                    y="Temperature"
                )

                st.plotly_chart(
                    fig1,
                    width="stretch",
                    key="temperature_trend_chart"
                )

            with chart_col2:

                st.subheader("Humidity Trend")

                hum_df = pd.DataFrame({
                    "Reading": list(range(len(humidity_history))),
                    "Humidity": list(humidity_history)
                })

                fig2 = px.line(
                    hum_df,
                    x="Reading",
                    y="Humidity"
                )

                st.plotly_chart(
                    fig2,
                    width="stretch",
                    key="humidity_trend_chart"
                )

            st.divider()

            st.subheader("🐛 Pest Detection Snapshot")

            if camera_manager is None:
                st.warning("Camera manager is not running")
            else:
                latest_frame = camera_manager.get_latest_frame()
                if latest_frame is not None:
                    st.image(frame_to_rgb(latest_frame), caption="Captured frame used for pest analysis", width="stretch")
                else:
                    st.info("Waiting for a captured frame...")

            if system_state["pest_detected"]:
                st.error(f"Pest detected: {system_state['pest_type']}")
            else:
                st.success("No pest detected")

            st.metric("Last Pest Check", system_state["last_pest_check"] or "Pending")

            if camera_manager is not None and getattr(camera_manager, "backend_window_error", None):
                st.warning(f"Backend window error: {camera_manager.backend_window_error}")

            st.divider()

        with camera_sidebar:
            st.subheader("📷 Camera Window")

            if camera_manager is None:
                st.warning("Camera manager unavailable")
            else:
                sidebar_frame = camera_manager.get_latest_frame()
                if sidebar_frame is not None:
                    st.image(frame_to_rgb(sidebar_frame), caption="Live camera window", width="stretch")
                else:
                    st.info("Waiting for camera window...")

        st.subheader("🧠 VLA Decision Engine")

        decisions = []

        if system_state["intrusion"]:
            decisions.append(
                "Intrusion detected → Alert farmer"
            )

        if system_state["pest_detected"]:
            decisions.append(
                f"Pest detected → Recommend pest control for {system_state['pest_type']}"
            )

        if temp > 35:
            decisions.append(
                "Temperature high → Monitor crops"
            )

        if humidity > 85:
            decisions.append(
                "Humidity high → Disease risk"
            )

        if len(decisions) == 0:
            decisions.append(
                "Normal conditions"
            )

        for d in decisions:
            st.write("•", d)

        if decision.get("esp32_alert", {}).get("active"):
            st.warning("ESP32 alert active: buzzer and light flashing")
        elif system_state["pest_detected"]:
            st.info("ESP32 alert command sent")
        else:
            st.caption("ESP32 pest alert is idle")

    except Exception as e:
        st.error(str(e))

    time.sleep(1)