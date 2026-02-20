import imageio_ffmpeg
import os
os.environ["PATH"] += os.pathsep + os.path.dirname(
    imageio_ffmpeg.get_ffmpeg_exe()
)

from dotenv import load_dotenv
load_dotenv()

import speech_recognition as sr
from gtts import gTTS
import pygame
import time
import re
from agent import chat_with_groq

# ── Speech Recognition ────────────────────────────────
recognizer = sr.Recognizer()
mic        = sr.Microphone()

print("Calibrating microphone...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
print("Microphone ready!")

# ── Pygame Audio ──────────────────────────────────────
pygame.mixer.init()

# ── Speak ─────────────────────────────────────────────
def speak(text):
    print(f"\nAGENT: {text}\n")
    try:
        tts        = gTTS(text=text, lang="en", slow=False)
        audio_path = os.path.join(os.getcwd(), "response.mp3")
        tts.save(audio_path)
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"TTS Error: {e}")

# ── Listen ────────────────────────────────────────────
def listen():
    with mic as source:
        print("🎤 Listening... (speak now)")
        try:
            audio = recognizer.listen(
                source, timeout=6, phrase_time_limit=8
            )
            text  = recognizer.recognize_google(audio)
            print(f"YOU SAID: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            print("No speech detected")
            return ""
        except sr.UnknownValueError:
            print("Could not understand")
            return ""
        except sr.RequestError as e:
            print(f"API error: {e}")
            return ""

# ── Extract Battery % from Speech ─────────────────────
def extract_battery(text):
    numbers = re.findall(r'\d+', text)
    for n in numbers:
        val = int(n)
        if 0 <= val <= 100:
            return val
    return None

# ── Main Loop ─────────────────────────────────────────
def main():
    speak(
        "Hello! I am Surya, your AI solar energy assistant. "
        "Tell me your battery level and I will help you "
        "plan your day!"
    )

    while True:
        try:
            text = listen()

            if not text:
                continue

            if any(w in text for w in ["stop", "exit", "quit", "bye"]):
                speak("Goodbye! Stay powered and safe!")
                break

            # Extract battery % if user mentioned it
            battery = extract_battery(text)

            # Send to Groq for intelligent response
            response, _, _ = chat_with_groq(text, battery_soc=battery)
            speak(response)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()