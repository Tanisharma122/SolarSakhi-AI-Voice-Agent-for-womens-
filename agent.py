import os
import re
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime
from tensorflow.keras.models import load_model
from groq import Groq
from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────
load_dotenv()
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "llama3-8b-8192")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ── Load LSTM Model ───────────────────────────────────
print("Loading model...")
model    = load_model("models/lstm_energy_brain.keras")
scaler_X = pickle.load(open("models/scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("models/scaler_y.pkl", "rb"))
df       = pd.read_csv(
    "data/ahmedabad_energy.csv",
    index_col="timestamp",
    parse_dates=True
)
print("Model loaded!")

# ── Groq Client ───────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Conversation Memory ───────────────────────────────
conversation_history = []

FEATURES = [
    "solar_irradiance", "temperature", "cloud_cover",
    "wind_speed", "humidity", "clearness_index",
    "usage_kwh", "is_daytime",
    "hour_sin", "hour_cos", "month_sin", "month_cos"
]

# ── Live Weather ──────────────────────────────────────
def get_live_weather():
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q=Ahmedabad&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=5).json()
        return {
            "temperature": r["main"]["temp"],
            "cloud_cover": r["clouds"]["all"],
            "wind_speed":  r["wind"]["speed"],
            "humidity":    r["main"]["humidity"],
        }
    except:
        # Fallback if no key or no internet
        return {
            "temperature": 32.0,
            "cloud_cover": 30.0,
            "wind_speed":  3.0,
            "humidity":    60.0,
        }

# ── LSTM Prediction ───────────────────────────────────
def predict_next_24hrs(battery_soc=None):
    past_24 = df[FEATURES].tail(24).values.copy()
    weather = get_live_weather()
    hour    = datetime.now().hour

    # Inject live weather into last row
    past_24[-1][1] = weather["temperature"]
    past_24[-1][2] = weather["cloud_cover"]
    past_24[-1][3] = weather["wind_speed"]
    past_24[-1][4] = weather["humidity"]
    past_24[-1][7] = 1 if 6 <= hour <= 18 else 0
    past_24[-1][8] = np.sin(2 * np.pi * hour / 24)
    past_24[-1][9] = np.cos(2 * np.pi * hour / 24)

    X           = scaler_X.transform(past_24)
    X           = X.reshape(1, 24, 12)
    pred_scaled = model.predict(X, verbose=0)
    pred        = scaler_y.inverse_transform(pred_scaled.reshape(24, 2))

    solar   = np.clip(pred[:, 0], 0, None)
    battery = np.clip(pred[:, 1], 0, 100)

    # Adjust battery if user told us their current level
    if battery_soc is not None:
        diff    = battery_soc - battery[0]
        battery = np.clip(battery + diff, 0, 100)

    return solar, battery

# ── Build Prediction Summary for Groq ────────────────
def get_prediction_summary(battery_soc=None):
    solar, battery    = predict_next_24hrs(battery_soc)
    weather           = get_live_weather()
    current_hour      = datetime.now().hour
    hours             = [(current_hour + i) % 24 for i in range(24)]

    peak_idx          = np.argmax(solar)
    peak_hour         = hours[peak_idx]
    peak_solar        = solar[peak_idx]
    good_hours        = [hours[i] for i in range(24) if solar[i] > 0.3]
    safety_risk_hours = [hours[i] for i in range(24) if battery[i] < 25]
    current_battery   = battery[0]
    total_savings     = solar.sum() * 8

    work_window = (
        f"{good_hours[0]}:00 to {good_hours[-1]}:00"
        if good_hours else "No good solar window today"
    )

    safety_status = (
        f"Risk after {safety_risk_hours[0]}:00"
        if safety_risk_hours else "Safe all night"
    )

    summary = f"""
LSTM PREDICTION SUMMARY (Real AI Model Output):
- Current battery: {current_battery:.0f}%
- Peak solar: {peak_hour}:00 with {peak_solar:.2f} kWh
- Best work window: {work_window}
- Night safety: {safety_status}
- Estimated savings: {total_savings:.0f} rupees
- Weather: {weather['temperature']}°C, clouds {weather['cloud_cover']}%
- Current time: {current_hour}:00
"""
    return summary, solar, battery

# ── Chat with Groq ────────────────────────────────────
def chat_with_groq(user_message, battery_soc=None):
    global conversation_history

    prediction_summary, solar, battery = get_prediction_summary(battery_soc)

    system_prompt = f"""You are Surya, a friendly AI solar energy assistant 
for women in rural India who use off-grid solar systems.

You speak in simple, warm, encouraging English.
You are like a helpful neighbor who understands energy.

You have real AI predictions from an LSTM neural network model:
{prediction_summary}

Your rules:
- Keep responses SHORT — maximum 2 to 3 sentences for voice
- Never use bullet points or lists — speak naturally
- Give specific actionable advice with exact times
- Always mention rupee savings when relevant
- If battery is below 25 percent, always warn about safety
- If user tells you their battery level, use that number
- Be encouraging and positive always
- Remember previous messages in conversation

Examples of good responses:
"Your battery looks good at 65 percent! Peak solar hits around 12 today, 
so run your sewing machine between 10 and 2 for free energy."

"I would wait until 11 AM to start cooking — that is when solar peaks. 
You could save around 30 rupees today by shifting to solar hours!"
"""

    conversation_history.append({
        "role": "user",
        "content": user_message
    })

    # Keep last 8 messages only for memory
    if len(conversation_history) > 8:
        conversation_history = conversation_history[-8:]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation_history
        ],
        max_tokens=120,
        temperature=0.7
    )

    assistant_message = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message, solar, battery

# ── Backward Compatible generate_advice ───────────────
def generate_advice(battery_soc_input=None):
    solar, battery    = predict_next_24hrs(battery_soc_input)
    current_hour      = datetime.now().hour
    hours             = [(current_hour + i) % 24 for i in range(24)]
    peak_idx          = np.argmax(solar)
    peak_hour         = hours[peak_idx]
    peak_solar        = solar[peak_idx]
    good_hours        = [hours[i] for i in range(24) if solar[i] > 0.3]
    safety_risk_hours = [hours[i] for i in range(24) if battery[i] < 25]
    current_battery   = battery[0]
    savings           = solar.sum() * 8
    parts             = []

    if current_battery > 70:
        parts.append(f"Battery at {current_battery:.0f}% — good level.")
    elif current_battery > 40:
        parts.append(f"Battery at {current_battery:.0f}% — moderate.")
    else:
        parts.append(f"Warning! Battery low at {current_battery:.0f}%.")

    if peak_solar > 0.2:
        parts.append(
            f"Peak solar at {peak_hour}:00 "
            f"with {peak_solar:.2f} kWh expected.")

    if good_hours:
        parts.append(
            f"Run appliances between "
            f"{good_hours[0]}:00 and {good_hours[-1]}:00.")

    if safety_risk_hours:
        parts.append(
            f"Battery risk after {safety_risk_hours[0]}:00. "
            f"Reserve power for lights.")
    else:
        parts.append("Battery safe through the night.")

    parts.append(f"Estimated savings: {savings:.0f} rupees.")
    return " ".join(parts), solar, battery


# ── Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Groq agent...")
    response, _, _ = chat_with_groq(
        "Hello! My battery is at 60 percent. "
        "When should I run my sewing machine today?"
    )
    print("\nGROQ RESPONSE:", response)