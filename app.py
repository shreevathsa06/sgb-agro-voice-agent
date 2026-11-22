import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from llm_responder import GroundedSalesLLM
from pydantic import BaseModel

from catalogue import CATALOGUE
from state_manager import SalesSessionManager

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ADAPTER_DIR = BASE_DIR / "sgb_qwen_qlora_adapter"

app = FastAPI(title="SGB Agro Industries Voice Sales Agent")

sales_llm = GroundedSalesLLM(ADAPTER_DIR)
session_manager = SalesSessionManager(llm=sales_llm)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TurnRequest(BaseModel):
    session_id: str = "judge-demo-1"
    text: Optional[str] = None
    language: Optional[str] = None
    audio_unclear: bool = False
    fast_mode: bool = True


class ResetRequest(BaseModel):
    session_id: str = "judge-demo-1"
    language: Optional[str] = "en"

class TtsRequest(BaseModel):
    text: str
    language: str = "en"


@app.get("/")
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "SGB Agro Voice Sales Agent API is running. Place UI in static/index.html."}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "adapter_exists": ADAPTER_DIR.exists(),
        "llm_loaded": sales_llm.loaded,
        "llm_error": sales_llm.load_error,
    }


@app.get("/api/catalogue")
def get_catalogue():
    return {"products": list(CATALOGUE.values())}


@app.post("/api/turn")
def voice_turn(req: TurnRequest):
    session = session_manager.get_session(req.session_id)
    if req.language in ("en", "kn"):
        session["language"] = req.language
    return session_manager.process_turn(
        session_id=req.session_id,
        user_text=req.text,
        audio_unclear=req.audio_unclear,
        fast_mode=req.fast_mode,
    )


@app.get("/api/summary/{session_id}")
def get_summary(session_id: str):
    return session_manager.build_summary(session_id)


@app.post("/api/reset")
def reset_session(req: ResetRequest):
    session = session_manager.reset_session(req.session_id, language=req.language)
    return session_manager._response_payload(session, reply="", status="reset")

@app.post("/api/tts")
def text_to_speech(req: TtsRequest):
    clean_text = (req.text or "").strip()
    if not clean_text:
        return {"error": "Empty text"}
    lang_code = "kn" if req.language == "kn" else "en"
    buf = io.BytesIO()
    tts = gTTS(text=clean_text, lang=lang_code)
    tts.write_to_fp(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")