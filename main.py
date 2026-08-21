import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="HiveAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
memory_hive: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    user_id: str
    message: Optional[str] = "" # <-- make it optional so clear works

class ChatResponse(BaseModel):
    reply: str
    history: List[dict]

SYSTEM_PROMPT = "You are HiveAI, a helpful and friendly assistant. Remember context from previous messages. Be concise."

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        user_id = req.user_id
        if user_id not in memory_hive:
            memory_hive[user_id] = []

        memory_hive[user_id].append({"role": "user", "content": req.message})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(memory_hive[user_id][-10:])

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        ai_reply = response.choices[0].message.content
        memory_hive[user_id].append({"role": "assistant", "content": ai_reply})

        return ChatResponse(reply=ai_reply, history=memory_hive[user_id])

    except Exception as e:
        print(f"ERROR: {e}") # check Render logs
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history")
def clear_history(req: ChatRequest): # <-- now accepts same ChatRequest
    if req.user_id in memory_hive:
        memory_hive[req.user_id] = []
    return {"status": "cleared", "user_id": req.user_id}