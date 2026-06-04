import serial
import json

ser = serial.Serial('COM6', 115200)

while True:
    try:
        line = ser.readline().decode().strip()

        data = json.loads(line)

        print(f"Temperature: {data['temp']}")
        print(f"Humidity: {data['humidity']}")
        print(f"PIR: {data['pir']}")
        print("-------------------")

    except Exception as e:
        print("Error:", e)