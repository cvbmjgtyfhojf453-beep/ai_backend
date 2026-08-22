import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from typing import List, Dict
import traceback

app = FastAPI(title="HiveAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Use IP + User-Agent as "user_id" since Android doesn't send one
memory_hive: Dict[str, List[dict]] = {}

SYSTEM_PROMPT = "You are HiveAI, a helpful and friendly assistant. Remember context from previous messages. Be concise."

def get_user_id(request: Request) -> str:
    # fallback: use client IP if no user_id sent
    client_ip = request.client.host if request.client else "unknown"
    return f"user_{client_ip}"

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        print("RECEIVED BODY:", body)

        message = body.get("message", "")
        user_id = get_user_id(request) # auto generate user_id

        if user_id not in memory_hive:
            memory_hive[user_id] = []

        # 1. Add user message
        memory_hive[user_id].append({"role": "user", "content": message})

        # 2. Build messages with system prompt + last 10
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(memory_hive[user_id][-10:])

        # 3. Call Groq
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        ai_reply = response.choices[0].message.content

        # 4. Save AI reply
        memory_hive[user_id].append({"role": "assistant", "content": ai_reply})

        return {"reply": ai_reply}

    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/clear-history")
async def clear_history(request: Request):
    try:
        user_id = get_user_id(request)
        if user_id in memory_hive:
            memory_hive[user_id] = []
        return {"status": "cleared"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})