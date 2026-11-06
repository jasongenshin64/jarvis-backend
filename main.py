import os
import json
import asyncio
from flask import Flask, request, jsonify, send_from_directory
from google import genai
from google.genai import types
import edge_tts

app = Flask(__name__)

AUDIO_FOLDER = "/tmp"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "JARVIS Server Online", "version": "1.1"})

# --- FITUR A: ROUTE INITIALIZATION (BOOTING SEQUENCE) ---
@app.route("/init", methods=["GET"])
def init():
    """Endpoint yang dipanggil ESP32 saat pertama kali dinyalakan (Power-On)"""
    text_reply = "All systems online, Sir. Helm controls are fully operational."
    audio_filename = "jarvis_init.wav"
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

    async def generate_voice():
        communicate = edge_tts.Communicate(text_reply, "en-GB-RyanNeural")
        await communicate.save(audio_path)

    try:
        asyncio.run(generate_voice())
    except Exception as e:
        print(f"Error TTS Init: {e}")

    base_url = request.host_url.rstrip("/")
    audio_url = f"{base_url}/static/{audio_filename}"

    return jsonify({
        "status": "READY",
        "jawaban": text_reply,
        "audio_url": audio_url
    })

# --- FITUR B: ROUTE CHAT (INTERAKTIF & KONTROL MCU STYLED) ---
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("text", "Hi Jarvis")
    user_text_lower = user_text.lower()

    # Prompt kepribadian J.A.R.V.I.S. ala MCU
    prompt = f"""
    Kamu adalah J.A.R.V.I.S., AI asisten canggih milik Tony Stark dari MCU.
    Pengguna berkata: "{user_text}".

    Aturan Respon:
    1. Jika pengguna meminta membuka topeng/helm -> "aksi": "BUKA"
    2. Jika meminta menutup topeng/helm -> "aksi": "TUTUP"
    3. Jika bertanya/ngobrol/perintah acak -> "aksi": "DIAM"
    4. Kepribadian: Sangat sopan, berkelas, cerdas, sedikit memiliki humor sarkas halus khas Jarvis MCU jika diajak bercanda, dan selalu panggil pengguna dengan 'Sir'.
    5. Panjang jawaban: MAKSIMAL 1 SAMPAI 2 KALIMAT PENDEK agar pemrosesan super cepat. Gunakan frasa khas seperti "Diagnostics complete", "As you wish, Sir", atau "Systems operational" jika relevan.

    Format JSON HANYA: {{"jawaban": "teks balasan", "aksi": "BUKA/TUTUP/DIAM"}}
    """

    action = "DIAM"
    text_reply = "At your service, Sir."
    ai_success = False

    if client:
        candidate_models = ['gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-flash-latest']
        for m_name in candidate_models:
            try:
                res = client.models.generate_content(
                    model=m_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=65,
                        temperature=0.3
                    )
                )
                if res and res.text:
                    raw_reply = res.text.strip()
                    if raw_reply.startswith("```json"):
                        raw_reply = raw_reply.replace("```json", "").replace("```", "").strip()
                    elif raw_reply.startswith("```"):
                        raw_reply = raw_reply.replace("```", "").strip()
                    
                    parsed_res = json.loads(raw_reply)
                    text_reply = parsed_res.get("jawaban", text_reply)
                    action = parsed_res.get("aksi", action)
                    ai_success = True
                    break
            except Exception:
                continue

    # Backup Offline jika AI High Demand (503)
    if not ai_success:
        if "buka" in user_text_lower and ("helm" in user_text_lower or "topeng" in user_text_lower):
            action = "BUKA"
            text_reply = "Opening the faceplate now, Sir."
        elif "tutup" in user_text_lower and ("helm" in user_text_lower or "topeng" in user_text_lower):
            action = "TUTUP"
            text_reply = "Closing the faceplate now, Sir."
        else:
            action = "DIAM"
            text_reply = "Systems operational, Sir, though external neural networks are busy."

    # Generate Suara Edge-TTS
    audio_filename = "jarvis_reply.wav"
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

    async def generate_voice():
        communicate = edge_tts.Communicate(text_reply, "en-GB-RyanNeural")
        await communicate.save(audio_path)

    try:
        asyncio.run(generate_voice())
    except Exception as e:
        print(f"Error TTS: {e}")

    base_url = request.host_url.rstrip("/")
    audio_url = f"{base_url}/static/{audio_filename}"

    return jsonify({
        "jawaban": text_reply,
        "aksi": action,
        "audio_url": audio_url
    })

@app.route("/static/<filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
