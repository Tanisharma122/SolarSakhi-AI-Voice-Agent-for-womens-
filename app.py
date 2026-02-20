import os
import numpy as np
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from agent import chat_with_groq, get_prediction_summary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    battery: Optional[float] = None

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response, solar, battery = chat_with_groq(
            req.message,
            battery_soc=req.battery
        )

        current_hour = datetime.now().hour
        hours        = [(current_hour + i) % 24 for i in range(24)]
        peak_hour    = int(hours[int(np.argmax(solar))])
        savings      = float(round(solar.sum() * 8, 0))
        battery_level = float(round(battery[0], 1))
        safe_tonight  = bool(float(battery.min()) >= 25)

        return {
            "response":      response,
            "battery_level": battery_level,
            "peak_hour":     peak_hour,
            "savings":       savings,
            "safe_tonight":  safe_tonight
        }

    except Exception as e:
        print(f"CHAT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response":      "Sorry I had trouble processing that. Please try again.",
            "battery_level": 50.0,
            "peak_hour":     12,
            "savings":       0.0,
            "safe_tonight":  True
        }

@app.get("/status")
def status():
    try:
        summary, solar, battery = get_prediction_summary()
        current_hour = datetime.now().hour
        hours        = [(current_hour + i) % 24 for i in range(24)]

        good_start = next((hours[i] for i in range(24) if solar[i] > 0.3), None)
        good_end   = next((hours[i] for i in range(23, -1, -1) if solar[i] > 0.3), None)

        return {
            "battery":      float(round(battery[0], 1)),
            "peak_hour":    int(hours[int(np.argmax(solar))]),
            "peak_solar":   float(round(solar.max(), 3)),
            "savings":      float(round(solar.sum() * 8, 0)),
            "safe_tonight": bool(float(battery.min()) >= 25),
            "good_hours": {
                "start": good_start,
                "end":   good_end
            }
        }
    except Exception as e:
        print(f"STATUS ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "battery":      50.0,
            "peak_hour":    12,
            "peak_solar":   0.5,
            "savings":      30.0,
            "safe_tonight": True,
            "good_hours":   {"start": 10, "end": 16}
        }