import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class GroundedSalesLLM:
    def __init__(self, adapter_dir: Path) -> None:
        self.adapter_dir = adapter_dir
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            use_cuda = torch.cuda.is_available()
            if use_cuda:
                try:
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    base_model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_NAME,
                        quantization_config=bnb_cfg,
                        device_map={"": 0},
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_NAME,
                        torch_dtype=torch.float16,
                        device_map={"": 0},
                        trust_remote_code=True,
                        local_files_only=True,
                    )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True,
                )

            if self.adapter_dir.exists() and (self.adapter_dir / "adapter_config.json").exists():
                self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
            else:
                self.model = base_model

            self.model.eval()
            self.loaded = True
            print("[GroundedSalesLLM] Model and adapter loaded successfully.")
        except Exception as exc:
            self.load_error = str(exc)
            self.loaded = False
            print(f"[GroundedSalesLLM] Load failed: {exc}")

    def generate_reply(
        self,
        user_text: str,
        language: str,
        session_state: Dict[str, Any],
        catalogue_data: Dict[str, Any],
        fallback_reply: str,
    ) -> str:
        if not self.loaded or self.model is None or self.tokenizer is None:
            return fallback_reply

        # For Kannada (kn), use the meticulously curated, native Kannada sales responses in fallback_reply.
        # This completely avoids small-model Indic degeneration / repetitive token loops (e.g. 'ಹೊಂದಿಗಳು ಹೊಂದಿಗಳು').
        if language == "kn":
            return fallback_reply

        lang_instruction = (
            "Respond ONLY in natural Kannada (kn). Do not reply in English sentences."
            if language == "kn"
            else "Respond ONLY in clear, helpful English (en)."
        )

        compact_catalogue = json.dumps(catalogue_data, ensure_ascii=False)
        compact_state = {
            "stage": session_state.get("stage"),
            "requirements": session_state.get("requirements"),
            "recommended_product_id": session_state.get("recommended_product_id"),
            "mock_order": session_state.get("mock_order"),
        }
        recent_turns: List[Dict[str, str]] = session_state.get("transcript", [])[-6:]

        system_msg = (
            "You are an SGB Agro Industries voice sales assistant. "
            "Analyze what the customer actually said in context of the conversation. "
            "CORE SALES RULES:\n"
            "1. When the customer asks for a grass cutting machine, NEVER recommend a specific product until you know:\n"
            "   - Whether they need a trolley attachment only (₹3,800 - ₹4,300) or a complete machine with engine (₹14,000 - ₹21,000).\n"
            "   - If full machine: warranty needed (none, 6 months, or 1 year) and their approximate budget.\n"
            "   - If trolley attachment: whether they have a sidepack or backpack brush cutter.\n"
            "2. When the customer greets midway in conversation, acknowledge the greeting politely and ask for the missing requirements for their enquiry.\n"
            "3. TARGET ACTION / GROUNDED RESPONSE: The grounded sales engine has determined this response:\n"
            f"   \"{fallback_reply}\"\n"
            "   You should deliver this message naturally and politely in the requested language without deviating or skipping questions.\n"
            "4. Use ONLY facts from GROUNDED_CATALOGUE for prices, engines, warranties, and attachments. Never invent specs.\n"
            f"{lang_instruction}\n\n"
            f"GROUNDED_CATALOGUE: {compact_catalogue}\n"
            f"SESSION_STATE: {json.dumps(compact_state, ensure_ascii=False)}"
        )

        messages = [{"role": "system", "content": system_msg}]
        for turn in recent_turns[:-1]:
            role = "assistant" if turn.get("speaker") == "agent" else "user"
            messages.append({"role": role, "content": turn.get("text", "")})
        messages.append({"role": "user", "content": user_text})

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if not reply:
                return fallback_reply

            # Safety check: if Kannada requested, ensure sufficient Kannada characters are present
            if language == "kn":
                kn_chars = sum(1 for ch in reply if "\u0C80" <= ch <= "\u0CFF")
                if kn_chars < 15:
                    return fallback_reply

            # Safety check: when qualifying grass cutting equipment (trolley vs full machine):
            # The reply MUST ask about both trolley AND complete machine with engine.
            req = session_state.get("requirements", {})
            if req.get("use_case") == "grass_weed_cutting" and req.get("owns_brush_cutter") is None:
                lower_reply = reply.lower()
                has_trolley = any(k in lower_reply for k in ["trolley", "attachment", "ಟ್ರಾಲಿ"])
                has_machine = any(k in lower_reply for k in ["full", "complete", "engine", "ಮಷಿನ್", "ಪೂರ್ಣ", "ಎಂಜಿನ್"])
                if not (has_trolley and has_machine):
                    return fallback_reply

            # Safety check: when user greets midway, ensure reply does not revert to generic initial welcome
            transcript = session_state.get("transcript", [])
            customer_turns = [t for t in transcript if t.get("speaker") == "customer"]
            if len(customer_turns) > 1 and any(w in fallback_reply.lower() for w in ["hello again", "we were discussing", "we were looking", "ಮತ್ತೊಮ್ಮೆ ನಮಸ್ಕಾರ", "ಮಾತನಾಡುತ್ತಿದ್ದೆವು"]):
                lower_reply = reply.lower()
                if "welcome to sgb agro" in lower_reply or "ಎಸ್‌ಜಿಬಿ ಅಗ್ರೋ ಇಂಡಸ್ಟ್ರೀಸ್‌ಗೆ ಸ್ವಾಗತ" in reply:
                    return fallback_reply

            return reply
        except Exception as exc:
            print(f"[GroundedSalesLLM] Generate failed: {exc}")
            return fallback_reply
