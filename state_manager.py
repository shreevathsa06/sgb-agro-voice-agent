import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from catalogue import (
    CATALOGUE,
    compare_brush_cutters,
    format_product_summary,
    get_product,
    recommend_product,
)

GREETING_PATTERNS_EN = re.compile(
    r"\b(hi|hello|hey|heya|namaste|namaskara|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you|howdy|bro)\b",
    re.IGNORECASE,
)

GREETING_TOKENS_KN = [
    "ನಮಸ್ಕಾರ", "ಶುಭೋದಯ", "ಶುಭ ಮಧ್ಯಾಹ್ನ", "ಶುಭ ಸಂಜೆ", "ಹೇಗಿದ್ದೀರಾ", "ಹಾಯ್", "ಹಲೋ", "ನಮಸ್ತೆ", "ಆರಾಮ",
]

PRODUCT_INTENT_KEYWORDS_EN = [
    "grass", "weed", "cut", "cutter", "cutting", "machine", "trolley", "attachment",
    "wheelbarrow", "wheel barrow", "barrow", "dumper", "soil", "manure", "compost",
    "carry", "transport", "load", "warranty", "budget", "price", "cost", "rupees",
    "order", "book", "buy", "purchase", "varsha", "143r", "walbro", "stable",
    "backpack", "sidepack", "engine", "compare", "difference", "vs", "sell", "catalogue", "catalog",
]

PRODUCT_INTENT_KEYWORDS_KN = [
    "ಹುಲ್ಲು", "ಕಳೆ", "ಕಟರ್", "ಕತ್ತರಿಸಲು", "ಕತ್ತರಿಸೋ", "ಕತ್ತರಿಸುವ", "ಮಷಿನ್", "ಟ್ರಾಲಿ", "ಅಟ್ಯಾಚ್‌ಮೆಂಟ್",
    "ವೀಲ್ ಬ್ಯಾರೋ", "ಬ್ಯಾರೋ", "ಡಂಪರ್", "ಮಣ್ಣು", "ಗೊಬ್ಬರ", "ಸಾಗಿಸಲು", "ಹೊರಲು", "ಸಾಗಾಟ",
    "ವಾರಂಟಿ", "ಬಜೆಟ್", "ಬೆಲೆ", "ರೂಪಾಯಿ", "ಆರ್ಡರ್", "ಬುಕ್", "ಖರೀದಿ", "ವರ್ಷ", "ವಾಲ್ಬ್ರೋ", "ಸ್ಟೇಬಲ್",
    "ಬ್ಯಾಕ್‌ಪ್ಯಾಕ್", "ಸೈಡ್‌ಪ್ಯಾಕ್", "ಎಂಜಿನ್", "ಹೋಲಿಕೆ", "ವ್ಯತ್ಯಾಸ", "ಏನೆಲ್ಲಾ", "ಕ್ಯಾಟಲಾಗ್", "ಉತ್ಪನ್ನಗಳು",
]

CATALOGUE_INQUIRY_WORDS = [
    "what do you sell", "what products", "show catalogue", "catalog", "catalogue",
    "list products", "what do you have", "who are you",
    "ಏನೆಲ್ಲಾ ಇದೆ", "ಯಾವ ಮಷಿನ್", "ಕ್ಯಾಟಲಾಗ್", "ಉತ್ಪನ್ನಗಳು", "ನೀವು ಯಾರು",
]

UNSUPPORTED_TOPICS = {
    "tractor": "ಟ್ರ್ಯಾಕ್ಟರ್",
    "ಟ್ರ್ಯಾಕ್ಟರ್": "ಟ್ರ್ಯಾಕ್ಟರ್",
    "harvester": "ಹಾರ್ವೆಸ್ಟರ್",
    "sprayer": "ಸ್ಪ್ರೇಯರ್",
    "ಸ್ಪ್ರೇಯರ್": "ಸ್ಪ್ರೇಯರ್",
    "water pump": "ವಾಟರ್ ಪಂಪ್",
    "ಪಂಪ್": "ವಾಟರ್ ಪಂಪ್",
    "emi": "ಇಎಂಐ / ಸಾಲ",
    "loan": "ಸಾಲ",
    "ಸಾಲ": "ಸಾಲ",
    "discount": "ಡಿಸ್ಕೌಂಟ್",
    "ಡಿಸ್ಕೌಂಟ್": "ಡಿಸ್ಕೌಂಟ್",
    "4-stroke": "4-ಸ್ಟ್ರೋಕ್ ಎಂಜಿನ್",
    "four stroke": "4-ಸ್ಟ್ರೋಕ್ ಎಂಜಿನ್",
    "4 stroke": "4-ಸ್ಟ್ರೋಕ್ ಎಂಜಿನ್",
    "refund": "ರಿಫಂಡ್",
}

CONFIRM_WORDS = [
    "yes", "confirm", "book it", "place order", "go ahead", "sure", "ok book", "okay book",
    "ಹೌದು", "ಬುಕ್ ಮಾಡಿ", "ಬುಕ್", "ಕನ್ಫರ್ಮ್", "ಸರಿ", "ಆರ್ಡರ್ ಮಾಡಿ", "ಮಾಡಿ",
]

CANCEL_CONFIRM_WORDS = [
    "no", "not now", "wait", "change", "cancel",
    "ಬೇಡ", "ಬದಲಾಯಿಸಬೇಕು", "ಬದಲಾವಣೆ", "ನಿಲ್ಲಿಸಿ",
]


class SalesSessionManager:
    def __init__(self, llm=None) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.llm = llm

    def create_session(self, session_id: Optional[str] = None, language: str = "en") -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())[:8]
        session = {
            "session_id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": language if language in ("en", "kn") else "en",
            "stage": "qualifying",
            "lead_status": "New - Qualifying",
            "requirements": {
                "use_case": None,
                "owns_brush_cutter": None,
                "brush_cutter_style": None,
                "needs_warranty": None,
                "budget_inr": None,
                "customer_name": None,
                "location": None,
            },
            "recommended_product_id": None,
            "misunderstandings": 0,
            "unsupported_flags": [],
            "transcript": [],
            "mock_order": None,
            "asked_barrow_type": False,
        }
        self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return self.create_session(session_id=session_id)
        return self.sessions[session_id]

    def reset_session(self, session_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        lang = language or self.get_session(session_id).get("language", "en")
        return self.create_session(session_id=session_id, language=lang)

    def _extract_budget(self, text: str) -> Optional[int]:
        clean = text.lower().replace(",", "")
        m_k = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:k|ಸಾವಿರ)\b", clean)
        if m_k:
            return int(float(m_k.group(1)) * 1000)
        m_num = re.search(r"\b(\d{4,6})\b", clean)
        if m_num:
            return int(m_num.group(1))
        return None

    def _detect_greeting(self, text: str) -> Tuple[bool, bool]:
        """
        Returns (has_greeting, is_pure_greeting).
        has_greeting: True if text contains greeting words (using word boundaries).
        is_pure_greeting: True if text is only a greeting/pleasantry and lacks product/order intent.
        """
        lower = text.lower()
        has_en_greet = bool(GREETING_PATTERNS_EN.search(lower))
        has_kn_greet = any(g in lower for g in GREETING_TOKENS_KN)
        has_greeting = has_en_greet or has_kn_greet

        has_en_intent = any(w in lower for w in PRODUCT_INTENT_KEYWORDS_EN)
        has_kn_intent = any(w in lower for w in PRODUCT_INTENT_KEYWORDS_KN)
        has_intent = has_en_intent or has_kn_intent or (self._extract_budget(text) is not None)

        is_pure = has_greeting and not has_intent
        return has_greeting, is_pure

    def _update_requirements(self, session: Dict[str, Any], text: str) -> List[str]:
        req = session["requirements"]
        lower = text.lower()
        updated_fields: List[str] = []

        transport_kw = [
            "soil", "manure", "compost", "barrow", "wheelbarrow", "wheel barrow", "dumper",
            "carry load", "transport", "tipping",
            "ಮಣ್ಣು", "ಗೊಬ್ಬರ", "ಬ್ಯಾರೋ", "ವೀಲ್ ಬ್ಯಾರೋ", "ಡಂಪರ್", "ಸಾಗಿಸಲು", "ಹೊರಲು", "ಸಾಗಾಟ",
        ]
        if any(w in lower for w in transport_kw):
            if req["use_case"] != "material_transport":
                req["use_case"] = "material_transport"
                updated_fields.append("use_case")

        cutting_kw = [
            "grass", "weed", "brush cutter", "brushcutter", "cutting", "cutter",
            "lawn mower", "mower", "trolley", "attachment", "varsha", "143r", "walbro",
            "ಹುಲ್ಲು", "ಕಳೆ", "ಕಟರ್", "ಬ್ರಷ್ ಕಟರ್", "ಕತ್ತರಿಸಲು", "ಕತ್ತರಿಸೋ", "ಕತ್ತರಿಸುವ", "ಟ್ರಾಲಿ", "ವರ್ಷ", "ವಾಲ್ಬ್ರೋ",
        ]
        if any(w in lower for w in cutting_kw):
            if req["use_case"] != "grass_weed_cutting":
                req["use_case"] = "grass_weed_cutting"
                updated_fields.append("use_case")

        # Explicit trolley attachment / already owns brush cutter
        owns_kw = [
            "already have", "already got", "already own", "have a brush cutter", "own a brush cutter",
            "have brush cutter", "only trolley", "trolley only", "trolley attachment", "just the trolley",
            "just trolley", "attachment only", "only attachment", "have one", "have a machine",
            "ಈಗಾಗಲೇ ಇದೆ", "ನನ್ನ ಬಳಿ ಇದೆ", "ಟ್ರಾಲಿ ಮಾತ್ರ", "ಕೇವಲ ಟ್ರಾಲಿ", "ಟ್ರಾಲಿ ಅಟ್ಯಾಚ್‌ಮೆಂಟ್", "ನಮ್ಮ ಬಳಿ ಇದೆ", "ಕಟರ್ ಇದೆ",
        ]
        # Wants complete/full machine with engine
        has_negation = any(neg in lower for neg in [
            "don't have", "do not have", "no brush cutter", "don't own",
            "ನನ್ನ ಬಳಿ ಇಲ್ಲ", "ಕಟರ್ ಇಲ್ಲ", "ಇಲ್ಲ", "ಹೊಸದು",
        ])
        full_kw = [
            "need full", "full machine", "complete machine", "with engine", "whole machine", "new machine",
            "complete set", "full set", "complete brush cutter", "full brush cutter",
            "ಪೂರ್ಣ", "ಹೊಸ ಮಷಿನ್", "ಎಂಜಿನ್ ಸಹಿತ", "ಸಂಪೂರ್ಣ ಮಷಿನ್", "ಪೂರ್ಣ ಮಷಿನ್",
        ]

        if any(w in lower for w in owns_kw) and not has_negation:
            if req["owns_brush_cutter"] is not True:
                req["owns_brush_cutter"] = True
                updated_fields.append("owns_brush_cutter")
        elif any(w in lower for w in full_kw) or has_negation:
            if req["owns_brush_cutter"] is not False:
                req["owns_brush_cutter"] = False
                updated_fields.append("owns_brush_cutter")

        # Style check for trolley
        if any(w in lower for w in ["backpack", "back pack", "ಬ್ಯಾಕ್‌ಪ್ಯಾಕ್", "ಬ್ಯಾಕ್ ಪ್ಯಾಕ್", "ಬ್ಯಾಕ್"]):
            req["brush_cutter_style"] = "backpack"
            req["owns_brush_cutter"] = True
            updated_fields.append("brush_cutter_style")
        elif any(w in lower for w in ["sidepack", "side pack", "ಸೈಡ್‌ಪ್ಯಾಕ್", "ಸೈಡ್ ಪ್ಯಾಕ್", "ಸೈಡ್"]):
            req["brush_cutter_style"] = "sidepack"
            req["owns_brush_cutter"] = True
            updated_fields.append("brush_cutter_style")

        # Warranty preferences
        if any(w in lower for w in [
            "1 year", "one year", "1-year", "1yr", "walbro", "3 hour", "3 hours",
            "ಒಂದು ವರ್ಷ", "1 ವರ್ಷ", "ವಾಲ್ಬ್ರೋ", "ಮೂರು ಗಂಟೆ", "3 ಗಂಟೆ",
        ]):
            req["needs_warranty"] = "1_year"
            req["owns_brush_cutter"] = False
            updated_fields.append("needs_warranty")
        elif any(w in lower for w in [
            "6 month", "6 months", "six month", "six months", "6-month", "stable", "half year",
            "ಆರು ತಿಂಗಳು", "6 ತಿಂಗಳು", "ಸ್ಟೇಬಲ್",
        ]):
            req["needs_warranty"] = "6_months"
            req["owns_brush_cutter"] = False
            updated_fields.append("needs_warranty")
        elif any(w in lower for w in ["no warranty", "without warranty", "warranty not needed", "zero warranty", "ವಾರಂಟಿ ಬೇಡ", "ವಾರಂಟಿ ಇಲ್ಲ"]):
            req["needs_warranty"] = False
            req["owns_brush_cutter"] = False
            updated_fields.append("needs_warranty")
        elif any(w in lower for w in ["warranty", "ವಾರಂಟಿ", "ವಾರಂಟಿ ಬೇಕು"]):
            if req["needs_warranty"] is None:
                req["needs_warranty"] = "6_months"
                req["owns_brush_cutter"] = False
                updated_fields.append("needs_warranty")

        budget = self._extract_budget(text)
        if budget is not None:
            req["budget_inr"] = budget
            if req["owns_brush_cutter"] is None and budget >= 12000:
                req["owns_brush_cutter"] = False
            updated_fields.append("budget_inr")

        return updated_fields

    def _next_discovery_question(self, session: Dict[str, Any], prepend_greeting: Optional[str] = None) -> str:
        req = session["requirements"]
        lang = session["language"]

        # Step 1: Discover primary use case
        if not req["use_case"]:
            if lang == "kn":
                q = "ನಮಸ್ಕಾರ, ಎಸ್‌ಜಿಬಿ ಅಗ್ರೋ ಇಂಡಸ್ಟ್ರೀಸ್‌ಗೆ ಸ್ವಾಗತ. ನಿಮಗೆ ಹುಲ್ಲು ಅಥವಾ ಕಳೆ ಕತ್ತರಿಸುವ ಮಷಿನ್ ಬೇಕೇ, ಅಥವಾ ಮಣ್ಣು ಮತ್ತು ಗೊಬ್ಬರ ಸಾಗಿಸಲು ವೀಲ್ ಬ್ಯಾರೋ ಬೇಕೇ?"
            else:
                q = "Hello and welcome to SGB Agro Industries. Are you looking for grass and weed cutting equipment, or a wheel barrow for carrying soil, manure, or farm produce?"
            return f"{prepend_greeting} {q}" if prepend_greeting else q

        # Step 2: Grass/weed cutting qualification steps
        if req["use_case"] == "grass_weed_cutting":
            # 2a. Ask if they already own a brush cutter (trolley only) or need complete machine with engine, warranty & budget
            if req["owns_brush_cutter"] is None:
                if lang == "kn":
                    q = (
                        "ಹುಲ್ಲು ಕತ್ತರಿಸುವ ಉಪಕರಣಗಳಲ್ಲಿ, ನಿಮ್ಮ ಬಳಿ ಈಗಾಗಲೇ ಬ್ರಷ್ ಕಟರ್ ಇದ್ದು ಕೇವಲ ಟ್ರಾಲಿ ಅಟ್ಯಾಚ್‌ಮೆಂಟ್ (₹3,800 - ₹4,300) ಬೇಕೇ, "
                        "ಅಥವಾ ಎಂಜಿನ್ ಸಹಿತ ಪೂರ್ಣ ಮಷಿನ್ (₹14,000 - ₹21,000) ಬೇಕೇ? ಹಾಗೆಯೇ ನಿಮ್ಮ ಅಂದಾಜು ಬಜೆಟ್ ಮತ್ತು ಎಂಜಿನ್ ವಾರಂಟಿ ಅಗತ್ಯವಿದೆಯೇ ಎಂದು ತಿಳಿಸಿ."
                    )
                else:
                    q = (
                        "For grass cutting, do you already own a brush cutter and just need a trolley attachment (₹3,800 - ₹4,300), "
                        "or do you need a complete machine with engine (₹14,000 - ₹21,000)? Also, please share your budget and if you need an engine warranty."
                    )
                return f"{prepend_greeting} {q}" if prepend_greeting else q

            # 2b. If they own a brush cutter, ask sidepack vs backpack
            if req["owns_brush_cutter"] is True and not req["brush_cutter_style"]:
                if lang == "kn":
                    q = "ನಿಮ್ಮ ಬಳಿ ಇರುವ ಬ್ರಷ್ ಕಟರ್ ಸೈಡ್‌ಪ್ಯಾಕ್ ಮಾಡೆಲ್ (ಟ್ರಾಲಿ ಬೆಲೆ ₹3,800) ಅಥವಾ ಬ್ಯಾಕ್‌ಪ್ಯಾಕ್ ಮಾಡೆಲ್ (ಟ್ರಾಲಿ ಬೆಲೆ ₹4,300)?"
                else:
                    q = "Is your existing brush cutter a sidepack model (SGB Sidepack Trolley - ₹3,800) or a backpack model (SGB Backpack Trolley - ₹4,300)?"
                return f"{prepend_greeting} {q}" if prepend_greeting else q

            # 2c. If they want a full machine, ask warranty & budget before recommending
            if req["owns_brush_cutter"] is False and req["budget_inr"] is None and req["needs_warranty"] is None:
                if lang == "kn":
                    q = (
                        "ಪೂರ್ಣ ಮಷಿನ್‌ನಲ್ಲಿ ನಿಮ್ಮ ಬಜೆಟ್ ಎಷ್ಟು ಮತ್ತು ನಿಮಗೆ ಎಂಜಿನ್ ವಾರಂಟಿ ಬೇಕೇ? "
                        "ನಮ್ಮಲ್ಲಿ ಎಸ್‌ಜಿಬಿ ವರ್ಷ ₹14,000 (ವಾರಂಟಿ ಇಲ್ಲ), 143R ಸ್ಟೇಬಲ್ ₹16,000 (6 ತಿಂಗಳ ವಾರಂಟಿ), ಮತ್ತು 143R ವಾಲ್ಬ್ರೋ ₹21,000 (1 ವರ್ಷದ ವಾರಂಟಿ) ಲಭ್ಯವಿದೆ."
                    )
                else:
                    q = (
                        "For a complete brush cutter, do you need an engine warranty (none, 6 months, or 1 year), and what approximate budget do you have in mind? "
                        "We have SGB Varsha at ₹14,000 (no warranty), 143R Stable at ₹16,000 (6 months warranty), and 143R WALBRO at ₹21,000 (1 year warranty)."
                    )
                return f"{prepend_greeting} {q}" if prepend_greeting else q

        # Step 3: Material transport qualification
        if req["use_case"] == "material_transport" and req["budget_inr"] is None and not session.get("asked_barrow_type"):
            session["asked_barrow_type"] = True
            if lang == "kn":
                q = "ನಿಮಗೆ ಸಾಮಾನ್ಯ ವೀಲ್ ಬ್ಯಾರೋ (₹7,500) ಬೇಕೇ ಅಥವಾ ಸುಲಭವಾಗಿ ಸುರಿಯುವ ಡಂಪರ್ ವೀಲ್ ಬ್ಯಾರೋ (₹8,500) ಬೇಕೇ?"
            else:
                q = "Would you prefer the SGB Normal Wheel Barrow (Rs. 7,500) for general transport, or the SGB Dumper Wheel Barrow (Rs. 8,500) with easy tipping for soil and manure?"
            return f"{prepend_greeting} {q}" if prepend_greeting else q

        # Step 4: Now we have enough details -> recommend the matching product
        product_id = recommend_product(req)
        if product_id:
            session["recommended_product_id"] = product_id
            session["stage"] = "recommended"
            session["lead_status"] = "Qualified - Product Recommended"
            prod = get_product(product_id)
            summary = format_product_summary(prod, lang=lang)
            if lang == "kn":
                rec = f"ನಿಮ್ಮ ಅಗತ್ಯಕ್ಕೆ ತಕ್ಕಂತೆ ನಾನು ಶಿಫಾರಸು ಮಾಡುವ ಉತ್ಪನ್ನ: {summary} ಈ ಮಾಕ್ ಆರ್ಡರ್ ಅನ್ನು ಬುಕ್ ಮಾಡಬೇಕೇ?"
            else:
                rec = f"Based on your requirements, I recommend {summary} Would you like me to confirm a mock order for this item?"
            return f"{prepend_greeting} {rec}" if prepend_greeting else rec

        if lang == "kn":
            q = "ನಿಮ್ಮ ಕೃಷಿ ಅಗತ್ಯ ಮತ್ತು ಬಜೆಟ್ ಎಷ್ಟು ಎಂದು ದಯವಿಟ್ಟು ತಿಳಿಸಿ."
        else:
            q = "Could you share your main farm requirement—cutting grass/weeds or transporting materials—and your budget?"
        return f"{prepend_greeting} {q}" if prepend_greeting else q

    def _build_midway_greeting_response(self, session: Dict[str, Any]) -> str:
        """
        Builds a polite contextual response when user greets midway through conversation,
        acknowledging the greet and prompting for current missing requirements.
        """
        lang = session["language"]
        req = session["requirements"]
        existing_pid = session.get("recommended_product_id")
        existing_prod = get_product(existing_pid)

        # Case 1: Product was already recommended
        if existing_prod:
            if lang == "kn":
                name_kn = existing_prod.get("name_kn", existing_prod["name"])
                return (
                    f"ಮತ್ತೊಮ್ಮೆ ನಮಸ್ಕಾರ! ನಾವು ಈ ಹಿಂದೆ {name_kn} (ಬೆಲೆ ₹{existing_prod['price_inr']:,}) ಬಗ್ಗೆ ಮಾತನಾಡಿದ್ದೆವು. "
                    f"ಈ ಮಾಕ್ ಆರ್ಡರ್ ಕನ್ಫರ್ಮ್ ಮಾಡಬೇಕೇ, ಅಥವಾ ನಿಮ್ಮ ಬಜೆಟ್ ಅಥವಾ ಅಗತ್ಯವನ್ನು ಬದಲಾಯಿಸಿ ಬೇರೆ ಉತ್ಪನ್ನ ನೋಡಬೇಕೇ?"
                )
            return (
                f"Hello again! Earlier we were looking at the {existing_prod['name']} (Rs. {existing_prod['price_inr']:,}). "
                f"Would you like to confirm a mock order for this, or adjust your requirements or budget?"
            )

        # Case 2: In grass cutting qualification
        if req.get("use_case") == "grass_weed_cutting":
            if req.get("owns_brush_cutter") is None:
                if lang == "kn":
                    return (
                        "ಮತ್ತೊಮ್ಮೆ ನಮಸ್ಕಾರ! ನಾವು ಹುಲ್ಲು ಕತ್ತರಿಸುವ ಮಷಿನ್ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೆವು. ನಿಮ್ಮ ಅಗತ್ಯಕ್ಕೆ ಸೂಕ್ತವಾದದ್ದನ್ನು ಆಯ್ಕೆ ಮಾಡಲು: "
                        "ನಿಮ್ಮ ಬಳಿ ಈಗಾಗಲೇ ಬ್ರಷ್ ಕಟರ್ ಇದ್ದು ಕೇವಲ ಟ್ರಾಲಿ ಅಟ್ಯಾಚ್‌ಮೆಂಟ್ (₹3,800 - ₹4,300) ಬೇಕೇ, ಅಥವಾ ಎಂಜಿನ್ ಸಹಿತ ಪೂರ್ಣ ಮಷಿನ್ (₹14,000 - ₹21,000) ಬೇಕೇ? "
                        "ಹಾಗೆಯೇ ನಿಮ್ಮ ಬಜೆಟ್ ಮತ್ತು ಎಂಜಿನ್ ವಾರಂಟಿ ಅಗತ್ಯ ತಿಳಿಸಿ."
                    )
                return (
                    "Hello again! We were discussing grass cutting machines. To help you choose the right option: "
                    "do you already own a brush cutter and just need a trolley attachment (₹3,800 - ₹4,300), or do you need a complete machine with engine (₹14,000 - ₹21,000)? "
                    "Also, please share your approximate budget and warranty preference."
                )
            if req.get("owns_brush_cutter") is True and not req.get("brush_cutter_style"):
                if lang == "kn":
                    return "ನಮಸ್ಕಾರ! ನಾವು ಬ್ರಷ್ ಕಟರ್ ಟ್ರಾಲಿ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೆವು. ನಿಮ್ಮ ಬಳಿ ಇರುವ ಕಟರ್ ಸೈಡ್‌ಪ್ಯಾಕ್ ಮಾಡೆಲ್ ಅಥವಾ ಬ್ಯಾಕ್‌ಪ್ಯಾಕ್ ಮಾಡೆಲ್ ಎಂದು ತಿಳಿಸಿ."
                return "Hello again! We were looking at brush cutter trolleys. Is your existing brush cutter a sidepack model or a backpack model?"

            if req.get("owns_brush_cutter") is False and req.get("needs_warranty") is None and req.get("budget_inr") is None:
                if lang == "kn":
                    return (
                        "ನಮಸ್ಕಾರ! ನಾವು ಪೂರ್ಣ ಬ್ರಷ್ ಕಟರ್ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೆವು. ನಿಮ್ಮ ಅಂದಾಜು ಬಜೆಟ್ ಎಷ್ಟು ಮತ್ತು "
                        "ನಿಮಗೆ ಎಂಜಿನ್ ವಾರಂಟಿ ಬೇಕೇ (ವಾರಂಟಿ ಇಲ್ಲ, 6 ತಿಂಗಳು, ಅಥವಾ 1 ವರ್ಷ) ಎಂದು ತಿಳಿಸಿ."
                    )
                return (
                    "Hello again! We were discussing complete brush cutters. What approximate budget do you have in mind, "
                    "and do you need an engine warranty (none, 6 months, or 1 year)?"
                )

        # Case 3: In material transport qualification
        if req.get("use_case") == "material_transport":
            if lang == "kn":
                return "ನಮಸ್ಕಾರ! ನಾವು ವೀಲ್ ಬ್ಯಾರೋ ಬಗ್ಗೆ ಮಾತನಾಡುತ್ತಿದ್ದೆವು. ನಿಮಗೆ ನಾರ್ಮಲ್ ವೀಲ್ ಬ್ಯಾರೋ (₹7,500) ಬೇಕೇ ಅಥವಾ ಸುಲಭವಾಗಿ ಸುರಿಯುವ ಡಂಪರ್ ವೀಲ್ ಬ್ಯಾರೋ (₹8,500) ಬೇಕೇ?"
            return "Hello again! We were discussing wheel barrows for transport. Would you prefer the Normal Wheel Barrow (₹7,500) or the Dumper Wheel Barrow (₹8,500)?"

        # Case 4: No use case yet
        if lang == "kn":
            return "ಮತ್ತೊಮ್ಮೆ ನಮಸ್ಕಾರ! ನಿಮ್ಮ ಕೃಷಿ ಕೆಲಸಕ್ಕೆ ಯಾವ ಉಪಕರಣ ಬೇಕು ಎಂದು ತಿಳಿಸಿ. ಹುಲ್ಲು ಕತ್ತರಿಸುವ ಮಷಿನ್ ಬೇಕೇ, ಅಥವಾ ಮಣ್ಣು/ಗೊಬ್ಬರ ಸಾಗಿಸಲು ವೀಲ್ ಬ್ಯಾರೋ ಬೇಕೇ?"
        return "Hello again! How can I assist you with your farming needs today? Are you looking for grass cutting machines or a wheel barrow for transport?"

    def _maybe_llm(self, cleaned: str, lang: str, session: Dict[str, Any], fallback_reply: str, fast_mode: bool = True) -> str:
        if fast_mode or lang == "kn":
            return fallback_reply
        if self.llm is not None:
            return self.llm.generate_reply(
                user_text=cleaned,
                language=lang,
                session_state=session,
                catalogue_data=CATALOGUE,
                fallback_reply=fallback_reply,
            )
        return fallback_reply

    def process_turn(self, session_id: str, user_text: Optional[str], audio_unclear: bool = False, fast_mode: bool = True) -> Dict[str, Any]:
        session = self.get_session(session_id)
        cleaned = (user_text or "").strip()
        lang = session["language"]

        # 1. Silence or unclear speech
        if audio_unclear or not cleaned:
            session["misunderstandings"] += 1
            session["transcript"].append({"speaker": "customer", "text": "[Unclear / Silent audio]"})
            if lang == "kn":
                reply = "ಕ್ಷಮಿಸಿ, ನಿಮ್ಮ ಮಾತು ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳುತ್ತೀರಾ?"
            else:
                reply = "Sorry, I didn't catch that clearly. Could you please repeat your requirement in English or Kannada?"
            session["transcript"].append({"speaker": "agent", "text": reply})
            return self._response_payload(session, reply, status="clarification_needed")

        # Auto language switch if Kannada characters detected
        if any("\u0C80" <= ch <= "\u0CFF" for ch in cleaned):
            session["language"] = "kn"
            lang = "kn"

        session["transcript"].append({"speaker": "customer", "text": cleaned})
        lower = cleaned.lower()

        has_greeting, is_pure_greeting = self._detect_greeting(cleaned)
        customer_turns = [t for t in session["transcript"][:-1] if t.get("speaker") == "customer"]
        is_midway = len(customer_turns) > 0

        # 2. Pure greeting handling (without other product keywords)
        if is_pure_greeting:
            if is_midway:
                reply = self._build_midway_greeting_response(session)
            else:
                if lang == "kn":
                    reply = "ನಮಸ್ಕಾರ! ಎಸ್‌ಜಿಬಿ ಅಗ್ರೋ ಇಂಡಸ್ಟ್ರೀಸ್‌ಗೆ ಸ್ವಾಗತ. ನಿಮಗೆ ಬ್ರಷ್ ಕಟರ್, ಟ್ರಾಲಿ ಅಥವಾ ವೀಲ್ ಬ್ಯಾರೋಗಳಲ್ಲಿ ಯಾವುದು ಬೇಕು?"
                else:
                    reply = "Good day! Welcome to SGB Agro Industries. How can I help you today—are you looking for a brush cutter, a trolley attachment, or a wheel barrow?"
            reply = self._maybe_llm(cleaned, lang, session, reply, fast_mode=fast_mode)
            session["transcript"].append({"speaker": "agent", "text": reply})
            return self._response_payload(session, reply, status="greeting")

        # 3. Catalogue overview / "what do you sell"
        if any(w in lower for w in CATALOGUE_INQUIRY_WORDS):
            if lang == "kn":
                reply = (
                    "ನಾವು ಎಸ್‌ಜಿಬಿ ಅಗ್ರೋ ಇಂಡಸ್ಟ್ರೀಸ್‌ನಲ್ಲಿ ಬ್ರಷ್ ಕಟರ್ ಟ್ರಾಲಿಗಳು (ಸೈಡ್‌ಪ್ಯಾಕ್ ₹3,800 ಮತ್ತು ಬ್ಯಾಕ್‌ಪ್ಯಾಕ್ ₹4,300), "
                    "ವೀಲ್ ಬ್ಯಾರೋಗಳು (ನಾರ್ಮಲ್ ₹7,500 ಮತ್ತು ಡಂಪರ್ ₹8,500), ಹಾಗೂ ಪೂರ್ಣ ಬ್ರಷ್ ಕಟರ್‌ಗಳು "
                    "(ವರ್ಷ ₹14,000, 143R ಸ್ಟೇಬಲ್ ₹16,000, ಮತ್ತು 143R ವಾಲ್ಬ್ರೋ ₹21,000) ಒದಗಿಸುತ್ತೇವೆ. ನಿಮಗೆ ಯಾವುದು ಬೇಕು?"
                )
            else:
                reply = (
                    "At SGB Agro Industries, we offer Brush Cutter Trolleys (Sidepack Rs. 3,800 and Backpack Rs. 4,300), "
                    "Wheel Barrows (Normal Rs. 7,500 and Dumper Rs. 8,500), and Complete Brush Cutters "
                    "(SGB Varsha Rs. 14,000, 143R Stable Rs. 16,000, and 143R WALBRO Rs. 21,000). Which one fits your farm work?"
                )
            reply = self._maybe_llm(cleaned, lang, session, reply, fast_mode=fast_mode)
            session["transcript"].append({"speaker": "agent", "text": reply})
            return self._response_payload(session, reply, status="catalogue_overview")

        # 4. Explicit order confirmation stage
        if session["stage"] == "awaiting_confirmation":
            if any(w in lower for w in CONFIRM_WORDS):
                prod = get_product(session["recommended_product_id"]) or CATALOGUE["sgb_143r_stable"]
                order_id = f"SGB-MOCK-{uuid.uuid4().hex[:6].upper()}"
                session["mock_order"] = {
                    "order_id": order_id,
                    "product_id": prod["id"],
                    "product_name": prod["name"],
                    "quantity": 1,
                    "amount_inr": prod["price_inr"],
                    "payment_status": "pending_or_not_applicable",
                    "order_type": "mock_order_simulation",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                session["stage"] = "booked"
                session["lead_status"] = "Mock Order Confirmed"
                if lang == "kn":
                    name_kn = prod.get("name_kn", prod["name"])
                    reply = (
                        f"ನಿಮ್ಮ ಮಾಕ್ ಆರ್ಡರ್ {order_id} ಕನ್ಫರ್ಮ್ ಆಗಿದೆ. "
                        f"ಉತ್ಪನ್ನ: {name_kn}, ಬೆಲೆ {prod['price_inr']} ರೂಪಾಯಿ. "
                        f"ಪಾವತಿ ಸ್ಥಿತಿ ಬಾಕಿ ಇದೆ ಎಂದು ದಾಖಲಿಸಲಾಗಿದೆ."
                    )
                else:
                    reply = (
                        f"Your mock order ({order_id}) is confirmed for {prod['name']} at Rs. {prod['price_inr']:,}. "
                        f"Payment status is marked as pending_or_not_applicable."
                    )
                session["transcript"].append({"speaker": "agent", "text": reply})
                return self._response_payload(session, reply, status="order_confirmed")

            if any(w in lower for w in CANCEL_CONFIRM_WORDS):
                session["stage"] = "qualifying"
                session["lead_status"] = "Qualifying - Requirement Updated"
                reply = (
                    "ಸರಿ, ನಿಮ್ಮ ಬಜೆಟ್ ಅಥವಾ ವಾರಂಟಿ ಅಗತ್ಯದಲ್ಲಿ ಯಾವ ಬದಲಾವಣೆ ಬೇಕು ಎಂದು ಹೇಳಿ."
                    if lang == "kn"
                    else "No problem. What would you like to change—budget, warranty, or another product?"
                )
                session["transcript"].append({"speaker": "agent", "text": reply})
                return self._response_payload(session, reply, status="updated")

        # 5. Unsupported / out-of-catalogue questions
        matched_keys = [k for k in UNSUPPORTED_TOPICS if k in lower]
        if matched_keys:
            session["unsupported_flags"].extend(matched_keys)
            if lang == "kn":
                topic_kn = ", ".join(dict.fromkeys(UNSUPPORTED_TOPICS[k] for k in matched_keys))
                reply = (
                    f"ಕ್ಷಮಿಸಿ, {topic_kn} ಕುರಿತ ಮಾಹಿತಿ ನಮ್ಮ ಎಸ್‌ಜಿಬಿ ಅಗ್ರೋ ಕ್ಯಾಟಲಾಗ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ, ಆದ್ದರಿಂದ ನಾನು ತಪ್ಪು ಮಾಹಿತಿ ನೀಡುವುದಿಲ್ಲ. "
                    f"ನಾನು ಬ್ರಷ್ ಕಟರ್ ಟ್ರಾಲಿ, ವೀಲ್ ಬ್ಯಾರೋ, ವರ್ಷ ಮತ್ತು 143R ಬ್ರಷ್ ಕಟರ್ ಬಗ್ಗೆ ನಿಖರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲೆ."
                )
            else:
                reply = (
                    f"I don't have information on '{', '.join(matched_keys)}' in the SGB Agro catalogue, "
                    f"so I won't guess. I can help with SGB brush cutter trolleys, wheel barrows, and Varsha or 143R brush cutters."
                )
            reply = self._maybe_llm(cleaned, lang, session, reply, fast_mode=fast_mode)
            session["transcript"].append({"speaker": "agent", "text": reply})
            return self._response_payload(session, reply, status="unsupported_acknowledged")

        # 6. Check for comparisons
        if any(w in lower for w in ["compare", "difference", "vs", "ಹೋಲಿಕೆ", "ವ್ಯತ್ಯಾಸ", "ಯಾವುದು ಉತ್ತಮ"]):
            reply = compare_brush_cutters(lang=lang)
            session["stage"] = "recommended"
            session["lead_status"] = "Comparing Options"
            reply = self._maybe_llm(cleaned, lang, session, reply, fast_mode=fast_mode)
            session["transcript"].append({"speaker": "agent", "text": reply})
            return self._response_payload(session, reply, status="comparison")

        # 7. Update requirements from user text
        updated = self._update_requirements(session, cleaned)

        # 8. Direct booking request: ONLY book if a product is already recommended or qualified
        is_order_intent = any(w in lower for w in ["order", "book", "buy", "purchase", "ಬುಕ್", "ಖರೀದಿ", "ಆರ್ಡರ್"])
        if is_order_intent:
            pid = session["recommended_product_id"] or recommend_product(session["requirements"])
            if pid:
                session["recommended_product_id"] = pid
                session["stage"] = "awaiting_confirmation"
                session["lead_status"] = "Awaiting Explicit Purchase Confirmation"
                prod = get_product(pid)
                if lang == "kn":
                    name_kn = prod.get("name_kn", prod["name"])
                    reply = f"ದಯವಿಟ್ಟು ಕನ್ಫರ್ಮ್ ಮಾಡಿ: {name_kn}, ಬೆಲೆ {prod['price_inr']} ರೂಪಾಯಿ ಗೆ ಮಾಕ್ ಆರ್ಡರ್ ಬುಕ್ ಮಾಡಬೇಕೇ?"
                else:
                    reply = (
                        f"Please confirm explicitly: should I book a mock order for {prod['name']} at Rs. {prod['price_inr']:,}? "
                        f"(Payment status remains pending_or_not_applicable.)"
                    )
                session["transcript"].append({"speaker": "agent", "text": reply})
                return self._response_payload(session, reply, status="awaiting_confirmation")
            # If not yet qualified, don't guess a product! Ask the qualification question below.

        # 9. Mid-call requirement change re-recommendation
        if updated and session["stage"] in ("recommended", "awaiting_confirmation", "booked"):
            new_pid = recommend_product(session["requirements"])
            if new_pid:
                session["recommended_product_id"] = new_pid
                session["stage"] = "awaiting_confirmation"
                session["lead_status"] = "Updated Requirement - Awaiting Confirmation"
                prod = get_product(new_pid)
                summary = format_product_summary(prod, lang=lang)
                if lang == "kn":
                    reply = f"ನಿಮ್ಮ ಹೊಸ ಅಗತ್ಯದ ಪ್ರಕಾರ ಬದಲಾಯಿಸಿದ್ದೇನೆ: {summary} ಈ ಮಾಕ್ ಆರ್ಡರ್ ಕನ್ಫರ್ಮ್ ಮಾಡಬೇಕೇ?"
                else:
                    reply = f"Updated for your latest requirement: {summary} Shall I confirm a mock order for this?"
                reply = self._maybe_llm(cleaned, lang, session, reply, fast_mode=fast_mode)
                session["transcript"].append({"speaker": "agent", "text": reply})
                return self._response_payload(session, reply, status="requirement_updated")

        # 10. If the turn had greeting + requirement, prepare polite greeting prefix
        prepend_greet = None
        if has_greeting:
            prepend_greet = "ನಮಸ್ಕಾರ!" if lang == "kn" else "Hello!"

        reply = self._next_discovery_question(session, prepend_greeting=prepend_greet)
        if session["stage"] == "recommended":
            session["stage"] = "awaiting_confirmation"
        reply = self._maybe_llm(cleaned, lang, session, reply)
        session["transcript"].append({"speaker": "agent", "text": reply})
        return self._response_payload(session, reply, status=session["stage"])

    def build_summary(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        prod = get_product(session["recommended_product_id"])
        return {
            "session_id": session["session_id"],
            "language": session["language"],
            "stage": session.get("stage", "qualifying"),
            "lead_status": session["lead_status"],
            "requirements": session["requirements"],
            "recommended_product": prod["name"] if prod else None,
            "recommended_product_id": session.get("recommended_product_id"),
            "misunderstandings_count": session["misunderstandings"],
            "unsupported_topics_flagged": list(dict.fromkeys(session["unsupported_flags"])),
            "turns_count": len(session["transcript"]),
            "mock_order": session["mock_order"],
            "transcript": session["transcript"],
        }

    def _response_payload(self, session: Dict[str, Any], reply: str, status: str) -> Dict[str, Any]:
        return {
            "session_id": session["session_id"],
            "status": status,
            "reply": reply,
            "language": session["language"],
            "stage": session.get("stage", "qualifying"),
            "lead_status": session["lead_status"],
            "requirements": session["requirements"],
            "recommended_product": get_product(session["recommended_product_id"]),
            "mock_order": session["mock_order"],
            "summary": self.build_summary(session["session_id"]),
        }
