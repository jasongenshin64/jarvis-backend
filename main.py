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
    return jsonify({"status": "JARVIS Server Online", "version": "1.0"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("text", "Hi Jarvis")

    prompt = f"""
    Kamu J.A.R.V.I.S. Pengguna berkata: "{user_text}".
    Aturan:
    1. Buka helm -> "aksi": "BUKA"
    2. Tutup helm -> "aksi": "TUTUP"
    3. Bicara biasa -> "aksi": "DIAM"
    4. Jawab sangat singkat (maksimal 1 kalimat), sopan, panggil 'Sir'.
    
    Format JSON HANYA: {{"jawaban": "teks", "aksi": "BUKA/TUTUP/DIAM"}}
    """

    action = "DIAM"
    text_reply = "At your service, Sir."

    if client:
        try:
            # Langsung kunci ke gemini-3.8-flash dengan respon cepat
            res = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=60,
                    temperature=0.2
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
        except Exception as e:
            text_reply = f"Gemini Error: {str(e)}"

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
