from dotenv import load_dotenv
import os
from fastapi.responses import JSONResponse
from typing import List, Dict
import traceback
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Request
from pydantic import BaseModel
from groq import Groq
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

app = FastAPI()
embed_model = None # Don't load on startup

def get_embed_model():
    global embed_model
    if embed_model is None:
        # This is the smallest model. 22MB
        embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
    return embed_model

load_dotenv()

app = FastAPI(title="ai_backend with Memory Hive")

# ========== CONFIG ==========
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

if not GROQ_API_KEY or not DATABASE_URL:
    raise ValueError("Missing GROQ_API_KEY or DATABASE_URL in environment variables")

client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2') # 384 dimension

# ========== DB SETUP ==========
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    register_vector(conn)
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Table 1: Full conversation history with embeddings
    cur.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    cur.execute('CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)')

    # Table 2: Memory Hive - Extracted facts about user
    cur.execute('''
        CREATE TABLE IF NOT EXISTS memory_hive (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE,
            facts JSONB DEFAULT '{}', -- {"name": "David", "location": "PH", "likes": "coding"}
            updated_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()
    print("Database + Memory Hive initialized")

# ========== MEMORY HIVE FUNCTIONS ==========
def update_memory_hive(user_id: str, user_message: str, ai_reply: str):
    """Extract facts from conversation and save to hive"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Get existing facts
    cur.execute('SELECT facts FROM memory_hive WHERE user_id = %s', (user_id,))
    row = cur.fetchone()
    facts = row[0] if row else {}

    # Simple extraction: look for "my name is", "I live in", "I like"
    msg_lower = user_message.lower()
    if "my name is" in msg_lower:
        facts["name"] = user_message.split("my name is")[-1].strip().split()[0]
    if "i live in" in msg_lower:
        facts["location"] = user_message.split("i live in")[-1].strip()
    if "i like" in msg_lower:
        facts["likes"] = user_message.split("i like")[-1].strip()

    # Upsert
    cur.execute('''
        INSERT INTO memory_hive (user_id, facts) VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET facts = %s, updated_at = NOW()
    ''', (user_id, json.dumps(facts), json.dumps(facts)))

    conn.commit()
    cur.close()
    conn.close()
    return facts

def get_memory_hive(user_id: str):
    """Get all stored facts about user"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT facts FROM memory_hive WHERE user_id = %s', (user_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else {}

# ========== CONVERSATION MEMORY FUNCTIONS ==========
def get_relevant_memories(user_id: str, query: str, limit: int = 12):
    conn = get_db_connection()
    cur = conn.cursor()
    query_embedding = embed_model.encode(query).tolist()

    cur.execute('''
        SELECT role, content FROM memories
        WHERE user_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s
    ''', (user_id, query_embedding, limit))

    memories = cur.fetchall()
    cur.close()
    conn.close()
    return [{"role": role, "content": content} for role, content in memories]

def save_memory(user_id: str, role: str, content: str):
    conn = get_db_connection()
    cur = conn.cursor()
    embedding = embed_model.encode(content).tolist()

    cur.execute('''
        INSERT INTO memories (user_id, role, content, embedding)
        VALUES (%s, %s, %s, %s)
    ''', (user_id, role, content, embedding))

    conn.commit()
    cur.close()
    conn.close()

# ========== API MODELS ==========
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# ========== CHAT ENDPOINT ==========
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        user_id = req.user_id
        user_message = req.message

        # 1. Get Memory Hive facts
        hive_facts = get_memory_hive(user_id)
        facts_str = json.dumps(hive_facts) if hive_facts else "No facts yet"

        # 2. Get relevant past conversation memories
        past_memories = get_relevant_memories(user_id, user_message)

        # 3. STRONG SYSTEM PROMPT with Memory Hive
        system_prompt = {
            "role": "system",
            "content": f"""You are an OBANOR with INFINITE MEMORY for each user.

            MEMORY HIVE - Known facts about this user: {facts_str}

            RULES:
            1. Always use the Memory Hive facts when relevant. Reference them naturally.
            2. Use the conversation history to stay consistent.
            3. If the user tells you something new about themselves, remember it forever.
            4. Be warm, personal, and recall details from previous chats.
            5. Never say "I don't have memory". You do.
            """
        }

        messages = [system_prompt]
        messages.extend(past_memories)
        messages.append({"role": "user", "content": user_message})

        # 4. Call Groq
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            temperature=0.8,
            max_tokens=1024
        )
        ai_reply = completion.choices[0].message.content

        # 5. Save to conversation memory + update Memory Hive
        save_memory(user_id, "user", user_message)
        save_memory(user_id, "assistant", ai_reply)
        update_memory_hive(user_id, user_message, ai_reply)

        return ChatResponse(reply=ai_reply)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "memory": "hive_active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)