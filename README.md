# SGB Agro Industries — AI Voice Sales Agent

A bilingual (English & Kannada) AI-powered voice sales agent for **SGB Agro Industries**, built with FastAPI, a fine-tuned Qwen 2.5 model, and a Bland.ai-inspired dark UI. It qualifies farm equipment leads through natural conversation and simulates mock order booking.

---

## 📸 Overview

| Feature | Detail |
|---|---|
| **Languages** | English (`en`) · Kannada (`kn`) |
| **Products** | 7 SGB Agro products across 3 categories |
| **Voice Input** | Browser Web Speech API (hands-free auto-send on silence) |
| **Voice Output** | Instant browser TTS (English) · Neural gTTS (Kannada & fallback) |
| **LLM** | Qwen 2.5-1.5B-Instruct + QLoRA adapter (runs locally, offline) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS (no framework) |

---

## 🗂️ Project Structure

```
sgb-voice-agent/
│
├── app.py                      # FastAPI application & API routes
├── state_manager.py            # Conversation state machine & session logic
├── catalogue.py                # Product catalogue + recommendation engine
├── llm_responder.py            # Qwen 2.5 QLoRA wrapper (grounded, offline)
├── train_qlora.py              # QLoRA fine-tuning script (training only)
├── sgb_sales_dataset.jsonl     # Training dataset (JSONL format)
├── training_metrics.json       # Training run metrics
│
├── sgb_qwen_qlora_adapter/     # Fine-tuned LoRA adapter weights (local)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
│
├── static/
│   └── index.html              # Complete single-file frontend UI
│
├── venv/                       # Python virtual environment
└── README.md
```

---

## ⚙️ System Requirements

### Hardware
| Component | Minimum | Recommended |
|---|---|---|
| **RAM** | 8 GB | 16 GB |
| **GPU** | — (CPU works) | CUDA-capable (4 GB+ VRAM) |
| **Storage** | 5 GB free | 10 GB free |
| **Internet** | Required for Kannada TTS (gTTS) | — |

### Software
| Requirement | Version |
|---|---|
| **OS** | Windows 10/11 · Linux · macOS |
| **Python** | 3.11.x |
| **CUDA** (optional) | 12.6 (for GPU inference) |
| **Browser** | Chrome 90+ or Edge 90+ (for Web Speech API) |

> [!NOTE]
> The LLM adapter runs fully **offline** — `HF_HUB_OFFLINE=1` is set in the code. No Hugging Face API key is needed at runtime. The base model (`Qwen/Qwen2.5-1.5B-Instruct`) must be pre-downloaded to the HuggingFace cache.

---

## 📦 Python Dependencies

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.165.0 | REST API framework |
| `uvicorn` | 0.54.0 | ASGI server |
| `pydantic` | 2.14.4 | Request/response validation |
| `starlette` | 1.10.0 | ASGI toolkit (FastAPI dep) |
| `gTTS` | 2.5.4 | Google Text-to-Speech (Kannada + Neural English) |
| `torch` | 2.14.1+cu126 | PyTorch (LLM inference) |
| `transformers` | 5.23.0 | Hugging Face model loading |
| `peft` | 0.22.0 | QLoRA / LoRA adapter loading |
| `bitsandbytes` | 0.50.2 | 4-bit quantization (NF4) |
| `accelerate` | 1.16.0 | Device mapping & mixed precision |
| `tokenizers` | 0.23.3 | Fast tokenizer for Qwen |
| `numpy` | 2.4.6 | Numerical ops |
| `scipy` | 1.17.1 | Scientific computing |
| `anyio` | 4.16.0 | Async I/O |
| `httpx2` | 2.16.2 | Async HTTP client |

---

## 🚀 How to Run

### 1. Clone / Navigate to the project

```powershell
cd C:\Users\LENOVO\sgb-voice-agent
```

### 2. Create and activate the virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install fastapi uvicorn pydantic gTTS torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft bitsandbytes accelerate tokenizers numpy scipy httpx anyio
```

> [!TIP]
> If you don't have a CUDA GPU, replace the `--index-url` torch install with the CPU build:
> ```powershell
> pip install torch torchvision torchaudio
> ```

### 4. Pre-download the base model (one-time)

```powershell
.\venv\Scripts\python.exe -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); print('Done')"
```

This downloads the Qwen 2.5-1.5B base model to your HuggingFace cache (~3 GB). Only needed once.

### 5. Start the server

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (dev mode) |
| `--port 8000` | Serve on port 8000 |

### 6. Open the UI

Open **Chrome or Edge** and navigate to:

```
http://localhost:8000
```

> [!IMPORTANT]
> Must use Chrome or Edge — Firefox does not support the Web Speech API (`SpeechRecognition`) used for voice input.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI (`static/index.html`) |
| `GET` | `/api/health` | Health check — LLM load status, adapter presence |
| `GET` | `/api/catalogue` | Returns full product catalogue as JSON |
| `GET` | `/api/summary/{session_id}` | Returns current session state & transcript |
| `POST` | `/api/turn` | Process a conversation turn (text input) |
| `POST` | `/api/reset` | Reset session to clean state |
| `POST` | `/api/tts` | Convert text to speech audio (MP3 stream via gTTS) |

### `/api/turn` request body

```json
{
  "session_id": "judge-demo-1",
  "text": "I need a grass cutting machine",
  "language": "en",
  "audio_unclear": false,
  "fast_mode": true
}
```

### `/api/turn` response

```json
{
  "session_id": "judge-demo-1",
  "status": "qualifying",
  "reply": "For grass cutting, do you already own a brush cutter and just need a trolley attachment...",
  "language": "en",
  "stage": "qualifying",
  "lead_status": "New - Qualifying",
  "requirements": { "use_case": "grass_weed_cutting", "owns_brush_cutter": null, ... },
  "recommended_product": null,
  "mock_order": null,
  "summary": { ... }
}
```

---

## 🧠 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Chrome/Edge)                   │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │  Web Speech API  │    │   index.html (Vanilla JS)    │   │
│  │  (Voice Input)   │───►│   - Mic auto-silence detect  │   │
│  │  SpeechRecog.    │    │   - Bubble transcript UI     │   │
│  └─────────────────┘    │   - 4-stage stepper          │   │
│                          │   - Requirement tiles        │   │
│  ┌─────────────────┐    │   - Product showcase card    │   │
│  │  speechSynthesis │◄───│   - Mock order receipt       │   │
│  │  (Instant TTS)   │    └──────────┬───────────────────┘   │
│  │  English only    │               │ fetch /api/turn        │
│  └─────────────────┘               │ fetch /api/tts         │
└───────────────────────────────────┼─────────────────────────┘
                                    │ HTTP (localhost:8000)
┌───────────────────────────────────▼─────────────────────────┐
│                     FastAPI Server (app.py)                  │
│                                                             │
│   /api/turn ──► SalesSessionManager.process_turn()          │
│   /api/tts  ──► gTTS(text, lang="kn") ──► MP3 stream        │
│   /api/reset──► SalesSessionManager.reset_session()         │
└───────────┬────────────────────────────────┬────────────────┘
            │                                │
┌───────────▼────────────┐    ┌─────────────▼──────────────────┐
│   state_manager.py     │    │   llm_responder.py              │
│                        │    │                                 │
│  - Greeting detection  │    │  Qwen2.5-1.5B-Instruct (base)  │
│  - Requirement extract │    │  + QLoRA SGB adapter (local)   │
│  - Stage machine       │    │  + 4-bit NF4 quantization      │
│  - Midway greet logic  │    │  + BitsAndBytes                 │
│  - Product recommend   │    │                                 │
│  - Mock order booking  │    │  fast_mode=True → skips LLM    │
└───────────┬────────────┘    │  (uses grounded responses)     │
            │                 └─────────────────────────────────┘
┌───────────▼────────────┐
│   catalogue.py         │
│                        │
│  7 products defined:   │
│  - Trolley Sidepack    │
│  - Trolley Backpack    │
│  - Normal Wheel Barrow │
│  - Dumper Wheel Barrow │
│  - SGB Varsha          │
│  - SGB 143R Stable     │
│  - SGB 143R WALBRO     │
│                        │
│  recommend_product()   │
│  format_product_summary│
└────────────────────────┘
```

---

## 🗣️ Conversation State Machine

The agent follows a strict qualification flow before recommending any product:

```
START
  │
  ▼
[1. DISCOVERY] ─── Ask: grass/weed cutting OR transport?
  │
  ├── Material Transport ──► Ask: Normal (₹7,500) or Dumper (₹8,500) barrow?
  │                               └──► RECOMMEND product
  │
  └── Grass/Weed Cutting
        │
        ▼
  [2. QUALIFY] ─── Ask: Already own a brush cutter?
        │
        ├── YES (owns brush cutter)
        │     └── Ask: Sidepack or Backpack model?
        │               ├── Sidepack ──► RECOMMEND Trolley Sidepack (₹3,800)
        │               └── Backpack ──► RECOMMEND Trolley Backpack (₹4,300)
        │
        └── NO (needs full machine)
              └── Ask: Budget + Warranty preference?
                        ├── No warranty / ≤₹14k ──► RECOMMEND SGB Varsha (₹14,000)
                        ├── 6 months / ≤₹16k   ──► RECOMMEND 143R Stable (₹16,000)
                        └── 1 year  / ≥₹21k    ──► RECOMMEND 143R WALBRO (₹21,000)
                                │
                                ▼
                          [3. MATCH] ─── Show product card, ask to confirm
                                │
                                ▼
                          [4. CONFIRMED] ─── Generate mock order receipt
```

---

## 🌐 Bilingual Support

| Feature | English | Kannada |
|---|---|---|
| **Voice Input** | Web Speech API (`en-IN`) | Web Speech API (`kn-IN`) |
| **Voice Output** | Browser speechSynthesis (< 50ms) | gTTS server call (~1–2s) |
| **Responses** | Grounded rule-based | Native Kannada strings (no LLM for kn) |
| **Auto-detect** | — | Kannada Unicode chars auto-switch language |
| **Greeting detect** | Regex word-boundary (`\bhi\b` etc.) | Unicode token match |

> [!NOTE]
> Kannada audio always uses Neural TTS (gTTS) because Chrome/Edge on Windows do not ship a `kn-IN` voice for the browser's Web Speech API. An internet connection is required for Kannada audio playback.

---

## 🔧 Configuration

| Setting | Location | Default |
|---|---|---|
| Session ID | UI header input | `judge-demo-1` |
| Language | UI pill (English / ಕನ್ನಡ) | `en` |
| Voice Engine | UI pill (⚡ Instant / Neural TTS) | `instant` |
| Port | `uvicorn` CLI | `8000` |
| Fast Mode | `TurnRequest.fast_mode` in `app.py` | `True` |
| LLM offline | `HF_HUB_OFFLINE=1` in `llm_responder.py` | Enabled |

---

## 🏋️ Fine-tuning (Optional)

The QLoRA adapter was trained on `sgb_sales_dataset.jsonl` using:

```powershell
.\venv\Scripts\python.exe train_qlora.py
```

| Training detail | Value |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Method** | QLoRA (4-bit NF4 quantization) |
| **LoRA rank** | 16 |
| **LoRA alpha** | 32 |
| **Target modules** | `q_proj`, `v_proj` |
| **Dataset** | `sgb_sales_dataset.jsonl` |
| **Output** | `sgb_qwen_qlora_adapter/` |

> [!CAUTION]
> Re-training overwrites the adapter. Back up `sgb_qwen_qlora_adapter/` before running `train_qlora.py`.

---

## 🐛 Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Kannada audio not playing | gTTS needs internet | Check network connection |
| English voice not speaking | Browser TTS blocked | Allow audio autoplay in browser settings |
| Mic button not working | Speech API not supported | Use Chrome or Edge (not Firefox/Safari) |
| LLM not loading | Base model not cached | Pre-download the model (Step 4 above) |
| `EPERM mkdir` error | Node/venv race condition | Re-run the same command once more |
| Server won't start on port 8000 | Port already in use | Use `--port 8001` or kill the existing process |
| gTTS `getaddrinfo failed` | No internet | Use Instant voice mode for English; Kannada audio won't work offline |

---

## 📄 License

Internal project — SGB Agro Industries. Not for public distribution.
