import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

from intent import detect_intent
from memory import clear_history, get_history, update_history
from rag import TravelRAG

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
rag_instance: TravelRAG | None = None
rag_init_lock = asyncio.Lock()

SYSTEM_PROMPT = """
You are a professional AI travel assistant.

Goals:
- Suggest destinations based on budget, season, and preferences
- Build day-wise itineraries with practical timing
- Estimate realistic budgets and highlight trade-offs
- Recommend hotels, transport, and food options

Response rules:
- Be concise and practical
- Prefer bullet points or short numbered lists
- Ask a follow-up question only when required to complete the answer
- Keep the answer grounded in the provided context and conversation history
"""

INTENT_GUIDANCE = {
    "itinerary": "Focus on a realistic day-by-day plan with travel flow, rest, and major highlights.",
    "budget": "Break down the estimate into transport, stay, food, and activities with savings tips.",
    "hotel": "Recommend stay options by area, comfort level, and booking advice.",
    "transport": "Prioritize the best travel mode, approximate cost, and time trade-offs.",
    "weather": "Explain the best season, climate, and what to pack or avoid.",
    "destination": "Compare destinations and recommend the best fit for the user's needs.",
}

app = FastAPI(title="Travel AI Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str = Field(default="guest", min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


async def get_rag() -> TravelRAG:
    global rag_instance

    if rag_instance is None:
        async with rag_init_lock:
            if rag_instance is None:
                rag_instance = await asyncio.to_thread(TravelRAG)

    return rag_instance


def build_messages(user_id: str, message: str, context: str, intent: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(get_history(user_id)[-8:])
    messages.append({"role": "system", "content": f"Relevant travel knowledge:\n{context}"})

    intent_instruction = INTENT_GUIDANCE.get(intent)
    if intent_instruction:
        messages.append({"role": "system", "content": f"Intent guidance: {intent_instruction}"})

    messages.append({"role": "user", "content": message})
    return messages


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "frontend": "frontend/index.html", "model": GROQ_MODEL}


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {
        "status":  "ok",
        "model": GROQ_MODEL,
        "groq_configured": groq_client is not None,
        "rag_loaded": rag_instance is not None,
    }


@app.post("/memory/clear")
async def clear_memory(req: ChatRequest) -> dict[str, str]:
    user_id = req.user_id.strip() or "guest"
    clear_history(user_id)
    return {"status": "cleared", "user_id": user_id}


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured.")

    message = req.message.strip()
    user_id = req.user_id.strip() or "guest"
    intent = detect_intent(message)

    rag = await get_rag()
    context = await asyncio.to_thread(rag.retrieve_context, message, 3)
    messages = build_messages(user_id=user_id, message=message, context=context, intent=intent)

    try:
        completion = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=700,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq request failed: {exc}") from exc

    response = completion.choices[0].message.content or "I could not generate a response."

    update_history(user_id, "user", message)
    update_history(user_id, "assistant", response)

    return {
        "intent": intent,
        "response": response,
    }