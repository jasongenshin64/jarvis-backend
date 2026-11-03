import os
import json
import asyncio
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import edge_tts

app = Flask(__name__)

AUDIO_FOLDER = "/tmp"

# Mengambil Gemini API Key dari Environment Variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "JARVIS Server Online", "version": "1.0"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("text", "Hi Jarvis")

    prompt = f"""
    Kamu adalah J.A.R.V.I.S. (Just A Rather Very Intelligent System).
    Pengguna berkata: "{user_text}".
    
    Aturan:
    1. Jika pengguna meminta membuka topeng/helm, tentukan "aksi": "BUKA".
    2. Jika meminta menutup topeng/helm, tentukan "aksi": "TUTUP".
    3. Jika hanya bicara biasa/bertanya, tentukan "aksi": "DIAM".
    4. Jawablah secara singkat (maksimal 2 kalimat), sopan, dan panggil pengguna dengan 'Sir'.
    
    Kembalikan respon HANYA berupa JSON valid dengan format:
    {{"jawaban": "teks respon", "aksi": "BUKA/TUTUP/DIAM"}}
    """

    action = "DIAM"
    text_reply = "At your service, Sir."

    if not GEMINI_API_KEY:
        text_reply = "API Key Gemini belum dipasang di Vercel, Sir."
    else:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            raw_reply = response.text.strip()
            
            if raw_reply.startswith("```json"):
                raw_reply = raw_reply.replace("```json", "").replace("```", "").strip()
            elif raw_reply.startswith("```"):
                raw_reply = raw_reply.replace("```", "").strip()
            
            parsed_res = json.loads(raw_reply)
            text_reply = parsed_res.get("jawaban", text_reply)
            action = parsed_res.get("aksi", action)
        except Exception as e:
            print(f"Error Gemini: {e}")
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
