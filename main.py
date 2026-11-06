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
    return jsonify({"status": "JARVIS Server Online", "version": "1.2"})

@app.route("/init", methods=["GET"])
def init():
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
    return jsonify({
        "status": "READY",
        "jawaban": text_reply,
        "audio_url": f"{base_url}/static/{audio_filename}"
    })

def process_gemini_response(res):
    text_reply = "At your service, Sir."
    action = "DIAM"
    if res and res.text:
        raw_reply = res.text.strip()
        if raw_reply.startswith("```json"):
            raw_reply = raw_reply.replace("```json", "").replace("```", "").strip()
        elif raw_reply.startswith("```"):
            raw_reply = raw_reply.replace("```", "").strip()
        try:
            parsed_res = json.loads(raw_reply)
            text_reply = parsed_res.get("jawaban", text_reply)
            action = parsed_res.get("aksi", action)
        except Exception:
            text_reply = raw_reply
    return text_reply, action

PROMPT_SYSTEM = """
Kamu adalah J.A.R.V.I.S., AI asisten canggih milik Tony Stark dari MCU.
Aturan:
1. Jika pengguna meminta membuka topeng/helm -> "aksi": "BUKA"
2. Jika meminta menutup topeng/helm -> "aksi": "TUTUP"
3. Jika bertanya/ngobrol biasa -> "aksi": "DIAM"
4. Kepribadian: Sopan, cerdas, sedikit humor sarkas halus khas MCU, panggil 'Sir'.
5. Jawab maksimal 1-2 kalimat pendek.

Format JSON HANYA: {"jawaban": "teks", "aksi": "BUKA/TUTUP/DIAM"}
"""

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("text", "Hi Jarvis")
    prompt = f"{PROMPT_SYSTEM}\nPengguna berkata: \"{user_text}\""

    text_reply = "At your service, Sir."
    action = "DIAM"

    if client:
        try:
            res = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=65, temperature=0.3)
            )
            text_reply, action = process_gemini_response(res)
        except Exception as e:
            text_reply = f"Gemini Error: {str(e)}"

    return respond_with_audio(text_reply, action)

@app.route("/chat-audio", methods=["POST"])
def chat_audio():
    """Endpoint untuk menerima raw WAV audio dari ESP32"""
    audio_bytes = request.data
    text_reply = "At your service, Sir."
    action = "DIAM"

    if client and audio_bytes:
        try:
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            res = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[PROMPT_SYSTEM, audio_part],
                config=types.GenerateContentConfig(max_output_tokens=65, temperature=0.3)
            )
            text_reply, action = process_gemini_response(res)
        except Exception as e:
            text_reply = f"Gemini Audio Error: {str(e)}"

    return respond_with_audio(text_reply, action)

def respond_with_audio(text_reply, action):
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
    return jsonify({
        "jawaban": text_reply,
        "aksi": action,
        "audio_url": f"{base_url}/static/{audio_filename}"
    })

@app.route("/static/<filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
