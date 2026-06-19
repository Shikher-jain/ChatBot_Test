from __future__ import annotations

import re


INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "itinerary": (
        r"\bitinerary\b",
        r"\bplan\b",
        r"\btrip plan\b",
        r"\bday[- ]?wise\b",
    ),
    "budget": (
        r"\bbudget\b",
        r"\bcost\b",
        r"\bprice\b",
        r"\bcheap\b",
        r"\bsaving\b",
    ),
    "hotel": (
        r"\bhotel\b",
        r"\bstay\b",
        r"\bresort\b",
        r"\baccommodation\b",
    ),
    "transport": (
        r"\bflight\b",
        r"\btrain\b",
        r"\bbus\b",
        r"\bcab\b",
        r"\btransport\b",
    ),
    "weather": (
        r"\bweather\b",
        r"\bseason\b",
        r"\bbest time\b",
        r"\bclimate\b",
    ),
    "destination": (
        r"\bdestination\b",
        r"\bwhere should i go\b",
        r"\brecommend\b",
        r"\bsuggest\b",
    ),
}


def detect_intent(message: str) -> str:
    normalized = message.lower()

    for intent, patterns in INTENT_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return intent

    return "general"