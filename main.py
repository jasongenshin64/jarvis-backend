import os
import json
import asyncio
import requests
import time
import io
import wave
import uuid
import traceback
import edge_tts
import miniaudio
from flask import Flask, request, jsonify
from vercel import blob

app = Flask(__name__)

# --- ENVIRONMENT VARIABLES ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo").strip()
LLM_MODEL = os.environ.get("GROQ_LLM_MODEL", "openai/gpt-oss-20b").strip() 
BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()

def requests_with_retry(url, method="POST", headers=None, json=None, data=None, files=None, timeout=15, max_retries=3):
    for attempt in range(max_retries):
        try:
            if method == "POST":
                response = requests.post(url, headers=headers, json=json, data=data, files=files, timeout=timeout)
            else:
                response = requests.get(url, headers=headers, timeout=timeout)

            if response.status_code in [429, 500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            return response
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    return response

async def generate_and_upload_wav(text_reply, filename):
    try:
        communicate = edge_tts.Communicate(text_reply, "en-GB-RyanNeural")
        mp3_buffer = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.extend(chunk["data"])

        if not mp3_buffer:
            raise RuntimeError("Audio kosong dari edge-tts")

        decoded = miniaudio.decode(
            bytes(mp3_buffer),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=16000
        )

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(decoded.nchannels)
            wf.setsampwidth(decoded.sample_width)
            wf.setframerate(decoded.sample_rate)
            wf.writeframes(decoded.samples.tobytes())
        
        wav_bytes = wav_buffer.getvalue()

        if not BLOB_TOKEN:
            raise Exception("BLOB_READ_WRITE_TOKEN belum diatur.")

        result = await blob.put_blob(
            filename, wav_bytes, access="public", content_type="audio/wav", add_random_suffix=True
        )
        return result.url

    except Exception as e:
        print(f"[TTS/BLOB ERROR] {e}")
        return None

def transcribe_audio(audio_bytes):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    
    if len(audio_bytes) > 5 * 1024 * 1024:
        raise Exception("Ukuran audio terlalu besar.")

    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": STT_MODEL, "language": "id", "response_format": "json"}

    response = requests_with_retry(url, method="POST", headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        raise Exception(f"Groq STT Error {response.status_code}: {response.text}")

def ask_jarvis_ai(user_text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Prompt yang lebih tegas sesuai saran ChatGPT
    prompt_system = """
    Kamu adalah J.A.R.V.I.S., asisten AI yang sopan, cerdas, dan memiliki kepribadian tenang seperti asisten profesional.
    Aturan:
    1. Panggil pengguna dengan sebutan Sir.
    2. Jika pengguna meminta membuka topeng atau helm, isi aksi dengan BUKA.
    3. Jika pengguna meminta menutup topeng atau helm, isi aksi dengan TUTUP.
    4. Jika pengguna bertanya atau mengobrol biasa, isi aksi dengan DIAM.
    5. Jawaban maksimal 1-2 kalimat pendek.
    6. Field jawaban harus berisi kalimat yang akan diucapkan.
    7. Field aksi hanya boleh BUKA, TUTUP, atau DIAM.
    Berikan jawaban sesuai format JSON yang ditentukan.
    """

    # MENERAPKAN STRICT JSON SCHEMA (Saran ChatGPT)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": user_text}
        ],
        "max_tokens": 150,
        "reasoning_effort": "low",
        "reasoning_format": "hidden",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "jarvis_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "jawaban": {"type": "string"},
                        "aksi": {
                            "type": "string",
                            "enum": ["BUKA", "TUTUP", "DIAM"]
                        }
                    },
                    "required": ["jawaban", "aksi"],
                    "additionalProperties": False
                }
            }
        }
    }

    print(f"[GROQ] Starting LLM request (Strict JSON)")
    response = requests_with_retry(url, method="POST", headers=headers, json=payload, timeout=25)
    
    print(f"[GROQ] HTTP Status: {response.status_code}")

    if response.status_code != 200:
        print(f"[GROQ] Response: {response.text[:1000]}")
        raise RuntimeError(f"Groq HTTP {response.status_code}: {response.text}")

    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return content

def process_ai_response(raw_text):
    text_reply = "At your service, Sir."
    action = "DIAM"
    if raw_text:
        try:
            parsed = json.loads(raw_text)
            text_reply = parsed.get("jawaban", text_reply)
            raw_action = parsed.get("aksi", "DIAM").upper()
            if raw_action in ["BUKA", "TUTUP", "DIAM"]:
                action = raw_action
        except Exception:
            text_reply = raw_text
    return text_reply, action

@app.route("/init", methods=["GET"])
def init():
    text_reply = "All systems online, Sir. Helm controls are fully operational."
    filename = f"init_{uuid.uuid4().hex}.wav"
    audio_url = asyncio.run(generate_and_upload_wav(text_reply, filename))
    if audio_url:
        return jsonify({"status": "READY", "jawaban": text_reply, "audio_url": audio_url})
    return jsonify({"status": "ERROR", "jawaban": "TTS Gagal"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("text", "Hi Jarvis")
    
    try:
        raw_ai_reply = ask_jarvis_ai(user_text)
        text_reply, action = process_ai_response(raw_ai_reply)
    except Exception as e:
        print(f"[CHAT ERROR] {repr(e)}")
        traceback.print_exc()
        text_reply = "Sistem saraf AI saya sedang sibuk, Sir."
        action = "DIAM"

    filename = f"reply_{uuid.uuid4().hex}.wav"
    audio_url = asyncio.run(generate_and_upload_wav(text_reply, filename))
    
    if audio_url:
        return jsonify({"jawaban": text_reply, "aksi": action, "audio_url": audio_url})
    return jsonify({"jawaban": "Gagal membuat suara", "aksi": "DIAM"}), 500

@app.route("/chat-audio", methods=["POST"])
def chat_audio():
    audio_bytes = request.data
    text_reply = "Maaf Sir, sensor suara saya sedang terganggu."
    action = "DIAM"

    if audio_bytes:
        try:
            user_text = transcribe_audio(audio_bytes)
            print(f"[STT] Pengguna: {user_text}")
            if user_text.strip() != "":
                raw_ai_reply = ask_jarvis_ai(user_text)
                text_reply, action = process_ai_response(raw_ai_reply)
        except Exception as e:
            print(f"[AUDIO ERROR] {repr(e)}")
            traceback.print_exc()
            text_reply = "Sistem saraf AI saya sedang sibuk, Sir."

    filename = f"reply_{uuid.uuid4().hex}.wav"
    audio_url = asyncio.run(generate_and_upload_wav(text_reply, filename))
    
    if audio_url:
        return jsonify({"jawaban": text_reply, "aksi": action, "audio_url": audio_url})
    return jsonify({"jawaban": "Gagal membuat suara", "aksi": "DIAM"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
