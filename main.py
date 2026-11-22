
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

# ==========================================
# ENVIRONMENT VARIABLES
# ==========================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
STT_MODEL = os.environ.get(
    "GROQ_STT_MODEL",
    "whisper-large-v3-turbo"
).strip()

LLM_MODEL = os.environ.get(
    "GROQ_LLM_MODEL",
    "openai/gpt-oss-20b"
).strip()

BLOB_TOKEN = os.environ.get(
    "BLOB_READ_WRITE_TOKEN",
    ""
).strip()


# ==========================================
# HTTP RETRY
# ==========================================

def requests_with_retry(
    url,
    method="POST",
    headers=None,
    json=None,
    data=None,
    files=None,
    timeout=15,
    max_retries=3
):
    response = None

    for attempt in range(max_retries):
        try:
            if method == "POST":
                response = requests.post(
                    url,
                    headers=headers,
                    json=json,
                    data=data,
                    files=files,
                    timeout=timeout
                )
            else:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout
                )

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


# ==========================================
# EDGE TTS + VERCEL BLOB
# ==========================================

async def generate_and_upload_wav(text_reply, filename):
    try:
        communicate = edge_tts.Communicate(
            text_reply,
            "en-GB-RyanNeural"
        )

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
            raise Exception(
                "BLOB_READ_WRITE_TOKEN belum diatur."
            )

        result = await blob.put_blob(
            filename,
            wav_bytes,
            access="public",
            content_type="audio/wav",
            add_random_suffix=True
        )

        return result.url

    except Exception as e:
        print(f"[TTS/BLOB ERROR] {repr(e)}")
        traceback.print_exc()
        return None


# ==========================================
# WHISPER SPEECH-TO-TEXT
# ==========================================

def transcribe_audio(audio_bytes):
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY belum diatur.")

    if not audio_bytes:
        return ""

    if len(audio_bytes) > 5 * 1024 * 1024:
        raise Exception("Ukuran audio terlalu besar.")

    url = (
        "https://api.groq.com/openai/v1/"
        "audio/transcriptions"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    files = {
        "file": (
            "audio.wav",
            audio_bytes,
            "audio/wav"
        )
    }

    data = {
        "model": STT_MODEL,
        "language": "id",
        "response_format": "json"
    }

    response = requests_with_retry(
        url,
        method="POST",
        headers=headers,
        files=files,
        data=data,
        timeout=30
    )

    print(f"[STT] HTTP Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        transcript = result.get("text", "")

        if not isinstance(transcript, str):
            transcript = ""

        return transcript.strip()

    raise Exception(
        f"Groq STT Error {response.status_code}: "
        f"{response.text[:1000]}"
    )


# ==========================================
# JARVIS LLM
# ==========================================

def ask_jarvis_ai(user_text):
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY belum diatur.")

    url = (
        "https://api.groq.com/openai/v1/"
        "chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt_system = """
    Kamu adalah J.A.R.V.I.S., asisten AI yang sopan,
    cerdas, dan memiliki kepribadian tenang seperti
    asisten profesional.

    Aturan:
    1. Panggil pengguna dengan sebutan Sir.
    2. Jika pengguna meminta membuka topeng atau helm,
       isi aksi dengan BUKA.
    3. Jika pengguna meminta menutup topeng atau helm,
       isi aksi dengan TUTUP.
    4. Jika pengguna bertanya atau mengobrol biasa,
       isi aksi dengan DIAM.
    5. Jawaban maksimal 1-2 kalimat pendek.
    6. Field jawaban harus berisi kalimat yang akan
       diucapkan.
    7. Field aksi hanya boleh BUKA, TUTUP, atau DIAM.
    8. Selalu jawab dalam bahasa yang digunakan pengguna.
    9. Berikan jawaban dalam format JSON yang diminta.
    """

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": prompt_system
            },
            {
                "role": "user",
                "content": user_text
            }
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
                        "jawaban": {
                            "type": "string"
                        },
                        "aksi": {
                            "type": "string",
                            "enum": [
                                "BUKA",
                                "TUTUP",
                                "DIAM"
                            ]
                        }
                    },
                    "required": [
                        "jawaban",
                        "aksi"
                    ],
                    "additionalProperties": False
                }
            }
        }
    }

    print("[GROQ] Starting LLM request...")

    response = requests_with_retry(
        url,
        method="POST",
        headers=headers,
        json=payload,
        timeout=25
    )

    print(f"[GROQ] HTTP Status: {response.status_code}")

    if response.status_code != 200:
        print(f"[GROQ] Response: {response.text[:1000]}")

        raise RuntimeError(
            f"Groq HTTP {response.status_code}: "
            f"{response.text}"
        )

    result = response.json()

    content = (
        result["choices"][0]["message"]["content"]
    )

    return content


# ==========================================
# PROCESS AI RESPONSE
# ==========================================

def process_ai_response(raw_text):
    text_reply = "At your service, Sir."
    action = "DIAM"

    if raw_text:
        try:
            parsed = json.loads(raw_text)

            text_reply = parsed.get(
                "jawaban",
                text_reply
            )

            raw_action = str(
                parsed.get("aksi", "DIAM")
            ).upper()

            if raw_action in ["BUKA", "TUTUP", "DIAM"]:
                action = raw_action

        except Exception:
            print("[LLM] JSON parsing failed.")
            text_reply = raw_text

    return text_reply, action


# ==========================================
# INIT ENDPOINT
# ==========================================

@app.route("/init", methods=["GET"])
def init():
    text_reply = (
        "All systems online, Sir. "
        "Helm controls are fully operational."
    )

    filename = f"init_{uuid.uuid4().hex}.wav"

    audio_url = asyncio.run(
        generate_and_upload_wav(
            text_reply,
            filename
        )
    )

    if audio_url:
        return jsonify({
            "status": "READY",
            "recognized": True,
            "transcript": "",
            "jawaban": text_reply,
            "aksi": "DIAM",
            "audio_url": audio_url
        })

    return jsonify({
        "status": "ERROR",
        "recognized": False,
        "transcript": "",
        "jawaban": "TTS Gagal",
        "aksi": "DIAM",
        "audio_url": ""
    }), 500


# ==========================================
# TEXT CHAT ENDPOINT
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_text = data.get("text", "Hi Jarvis")

    try:
        raw_ai_reply = ask_jarvis_ai(user_text)

        text_reply, action = process_ai_response(
            raw_ai_reply
        )

    except Exception as e:
        print(f"[CHAT ERROR] {repr(e)}")
        traceback.print_exc()

        text_reply = (
            "Sistem saraf AI saya sedang sibuk, Sir."
        )

        action = "DIAM"

    filename = f"reply_{uuid.uuid4().hex}.wav"

    audio_url = asyncio.run(
        generate_and_upload_wav(
            text_reply,
            filename
        )
    )

    if audio_url:
        return jsonify({
            "recognized": True,
            "transcript": user_text,
            "jawaban": text_reply,
            "aksi": action,
            "audio_url": audio_url
        })

    return jsonify({
        "recognized": True,
        "transcript": user_text,
        "jawaban": "Gagal membuat suara",
        "aksi": "DIAM",
        "audio_url": ""
    }), 500


# ==========================================
# AUDIO CHAT ENDPOINT - ESP32
# ==========================================

@app.route("/chat-audio", methods=["POST"])
def chat_audio():
    audio_bytes = request.data

    if not audio_bytes:
        print("[AUDIO] Empty audio received.")

        return jsonify({
            "recognized": False,
            "transcript": "",
            "jawaban": "",
            "aksi": "DIAM",
            "audio_url": ""
        })

    # --------------------------------------
    # SPEECH RECOGNITION
    # --------------------------------------

    try:
        user_text = transcribe_audio(audio_bytes)

        print(f"[STT] Transcript: {user_text}")

    except Exception as e:
        print(f"[STT ERROR] {repr(e)}")
        traceback.print_exc()

        return jsonify({
            "recognized": False,
            "transcript": "",
            "jawaban": "",
            "aksi": "DIAM",
            "audio_url": ""
        })

    # --------------------------------------
    # NO SPEECH DETECTED
    # --------------------------------------

    if not user_text.strip():
        print("[STT] No speech recognized. Silent response.")

        return jsonify({
            "recognized": False,
            "transcript": "",
            "jawaban": "",
            "aksi": "DIAM",
            "audio_url": ""
        })

    # --------------------------------------
    # AI RESPONSE
    # --------------------------------------

    try:
        raw_ai_reply = ask_jarvis_ai(user_text)

        text_reply, action = process_ai_response(
            raw_ai_reply
        )

    except Exception as e:
        print(f"[AI ERROR] {repr(e)}")
        traceback.print_exc()

        text_reply = (
            "Sistem saraf AI saya sedang sibuk, Sir."
        )

        action = "DIAM"

    # --------------------------------------
    # TEXT TO SPEECH
    # --------------------------------------

    filename = f"reply_{uuid.uuid4().hex}.wav"

    audio_url = asyncio.run(
        generate_and_upload_wav(
            text_reply,
            filename
        )
    )

    if audio_url:
        print("[AUDIO] Response successfully generated.")

        return jsonify({
            "recognized": True,
            "transcript": user_text,
            "jawaban": text_reply,
            "aksi": action,
            "audio_url": audio_url
        })

    print("[AUDIO] TTS generation failed.")

    return jsonify({
        "recognized": True,
        "transcript": user_text,
        "jawaban": text_reply,
        "aksi": action,
        "audio_url": ""
    }), 500


# ==========================================
# LOCAL SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
