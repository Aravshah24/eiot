import serial
import json
import pandas as pd
from datetime import datetime
import os

ser = serial.Serial('COM6', 115200)

csv_file = "farm_data.csv"

while True:
    try:
        line = ser.readline().decode().strip()

        data = json.loads(line)

        print(data)

        row = {
            "time": datetime.now(),
            "temperature": data["temp"],
            "humidity": data["humidity"],
            "pir": data["pir"]
        }

        df = pd.DataFrame([row])

        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)

    except Exception as e:
        print("Error:", e)