import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from typing import List, Dict
import traceback

app = FastAPI(title="HiveAI Backend")

# Allow Android app to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# MEMORY HIVE - RAM storage. Resets on Render restart
memory_hive: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    history: List[dict]

SYSTEM_PROMPT = """You are HiveAI, a helpful and friendly assistant.
Remember context from previous messages.
Be concise and conversational.
"""

@app.get("/")
def health():
    return {"status": "ok", "model": "openai/gpt-oss-20b"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        user_id = req.user_id

        # 1. Init memory for new user
        if user_id not in memory_hive:
            memory_hive[user_id] = []

        # 2. Add user message to memory
        memory_hive[user_id].append({"role": "user", "content": req.message})

        # 3. Build messages: System + Last 10 messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(memory_hive[user_id][-10:])

        # 4. Call Groq
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        ai_reply = response.choices[0].message.content

        # 5. Save AI reply to memory
        memory_hive[user_id].append({"role": "assistant", "content": ai_reply})

        return ChatResponse(reply=ai_reply, history=memory_hive[user_id])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
def get_history(user_id: str):
    """Get full chat history for a user"""
    return {"user_id": user_id, "history": memory_hive.get(user_id, [])}

@app.post("/clear-history")
def clear_history(req: ChatRequest):
    """Clear memory for a user"""
    if req.user_id in memory_hive:
        memory_hive[req.user_id] = []
    return {"status": "cleared", "user_id": req.user_id}