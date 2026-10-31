from fastapi.responses import JSONResponse
from typing import List, Dict
import traceback
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import bcrypt
import httpx
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import sessionmaker
import os, json, uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request, BackgroundTasks, File, Form
# from fastapi.security import OAuth2PasswordBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from groq import Groq
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from docx import Document
from jose import jwt, JWTError
from passlib.context import CryptContext #52 Auth
import requests
from apscheduler.schedulers.background import BackgroundScheduler #22 Proactive
from google.oauth2.credentials import Credentials #28 Calendar
from googleapiclient.discovery import build
# from playwright.sync_api import sync_playwright #37 Browser
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from fastapi import Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import io
import time
from collections import defaultdict
from io import BytesIO
import fitz  # PyMuPDF
import re
import urllib.parse

load_dotenv()

app = FastAPI(title="AI OS Backend - Tier 1-8")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SECRET_KEY = os.getenv("SECRET_KEY", "secretkeyokay")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30 # 30 day
# oauth2 = OAuth2PasswordBearer(tokenUrl="token")
pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer()  # instead of OAuth2PasswordBearer
scheduler = BackgroundScheduler()

# DB + VECTOR - Auto create tables
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DATABASE_URL = os.getenv("DATABASE_URL")




FALLBACK_UNTIL = None
LAST_FALLBACK_LOG = None

def groq_chat(messages, max_tokens=300):
    global FALLBACK_UNTIL, LAST_FALLBACK_LOG
    
    # If 20b is rate-limited, don't spam trying it
    models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    if FALLBACK_UNTIL and datetime.utcnow() < FALLBACK_UNTIL:
        models = ["openai/gpt-oss-120b"] # skip 20b for now

    for model in models:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            FALLBACK_UNTIL = None # reset on success
            return res
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["429", "413", "rate_limit", "too large", "decommissioned", "404", "access"]):
                # log only once per hour to avoid spam
                now = datetime.utcnow()
                if not LAST_FALLBACK_LOG or (now - LAST_FALLBACK_LOG).seconds > 3600:
                    print(f"{model} failed ({str(e)[:120]}), trying next...")
                    LAST_FALLBACK_LOG = now
                
                if "20b" in model and "429" in err:
                    FALLBACK_UNTIL = now + timedelta(hours=1)
                continue
            raise e
    
    raise Exception("Both gpt-oss models hit daily limit. Resets 00:00 UTC (1am Lagos). Upgrade at console.groq.com/settings/billing")



def fix_embedding_dim():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        # Try to alter first
        cur.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE vector(1024);")
        print("✅ Embedding dim updated to 1024")
    except Exception as e:
        print(f"⚠️ Alter failed: {e}")
        conn.rollback()
        print("Dropping and recreating table...")
        cur.execute("DROP TABLE IF EXISTS memories;")
        conn.commit()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        # Now recreate table with correct dims
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (id UUID PRIMARY KEY, user_id UUID, content TEXT, embedding vector(1024), category TEXT, importance FLOAT, emotion TEXT, privacy_mode BOOLEAN);
        """)
        print("✅ Table recreated with 1024 dims")
    
    conn.commit()
    cur.close()
    conn.close()

# Call it on startup
fix_embedding_dim()

class ChatRequest(BaseModel):
    message: str
    persona: str = "assistant"
    roast_level: int = 0

def get_db():
    return psycopg2.connect(DATABASE_URL) # sslmode='require')


def create_tables():
    conn = psycopg2.connect(DATABASE_URL) # sslmode='require')
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    sql = """
    -- 1. EXTENSION FIRST
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. TABLES SECOND  
-- CREATE TABLE IF NOT EXISTS users (id UUID PRIMARY KEY, email TEXT UNIQUE, password_hash TEXT);
CREATE TABLE IF NOT EXISTS memories (id UUID PRIMARY KEY, user_id UUID, content TEXT, embedding vector(1536), category TEXT, importance FLOAT, emotion TEXT, privacy_mode BOOLEAN);
CREATE TABLE IF NOT EXISTS memory_hive (user_id UUID PRIMARY KEY, summary JSONB);
CREATE TABLE IF NOT EXISTS tasks (id UUID PRIMARY KEY, user_id UUID, title TEXT, due_at TIMESTAMP, done BOOLEAN);
CREATE TABLE IF NOT EXISTS contacts (id UUID PRIMARY KEY, user_id UUID, name TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS webhooks (id UUID PRIMARY KEY, user_id UUID, trigger TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS backups (id UUID PRIMARY KEY, created_at TIMESTAMP, url TEXT);


-- ENABLE UUID EXTENSION
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- TIER 1: USERS + PROFILES
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    full_name TEXT,
    avatar_url TEXT,
    bio TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- TIER 2: FRIENDS + BLOCKS
CREATE TABLE IF NOT EXISTS friendships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    friend_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending','accepted','blocked')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, friend_id),
    CHECK (user_id != friend_id)
);

-- TIER 3: POSTS
CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- TIER 4: COMMENTS + LIKES
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id, user_id)
);

-- TIER 5: CHATS + MESSAGES
CREATE TABLE IF NOT EXISTS chats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    is_group BOOLEAN DEFAULT FALSE,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_participants (
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    sender_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- TIER 6-8: INDEXES FOR PERFORMANCE
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_likes_post_id ON likes(post_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_friendships_user_id ON friendships(user_id);
CREATE INDEX IF NOT EXISTS idx_friendships_friend_id ON friendships(friend_id);
CREATE INDEX IF NOT EXISTS idx_friendships_status ON friendships(status);

-- TIER 7: AUTO UPDATE updated_at TRIGGER
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_timestamp_users ON users;
DROP TRIGGER IF EXISTS set_timestamp_profiles ON profiles;
DROP TRIGGER IF EXISTS set_timestamp_posts ON posts;
DROP TRIGGER IF EXISTS set_timestamp_comments ON comments;
DROP TRIGGER IF EXISTS set_timestamp_chats ON chats;
DROP TRIGGER IF EXISTS set_timestamp_friendships ON friendships;

CREATE TRIGGER set_timestamp_users BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();
CREATE TRIGGER set_timestamp_profiles BEFORE UPDATE ON profiles FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();
CREATE TRIGGER set_timestamp_posts BEFORE UPDATE ON posts FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();
CREATE TRIGGER set_timestamp_comments BEFORE UPDATE ON comments FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();
CREATE TRIGGER set_timestamp_chats BEFORE UPDATE ON chats FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();
CREATE TRIGGER set_timestamp_friendships BEFORE UPDATE ON friendships FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();

    """
    cur.execute(sql)
    cur.close()
    conn.close()
    print("Tables created/verified ✅")

create_tables()

# DB + VECTOR
#conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode='require')
#register_vector(conn)
#cur = conn.cursor()

# CLOUDINARY
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUD_KEY"),
    api_secret=os.getenv("CLOUD_SECRET")
)

# ====== AUTH #52 ======
@app.post("/register")
@limiter.limit("5/minute")
def register(
    request: Request,
    email: str = Body(...),
    password: str = Body(...)
):
    hash = pwd.hash(password)
    user_id = str(uuid.uuid4())
    
    conn = get_db()  # <-- Must use get_db() here
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (id,email,password_hash) VALUES (%s,%s,%s)", (user_id, email, hash))
            conn.commit()
        return {"user_id": user_id, "message": "User created successfully"}
    except:
        conn.rollback()
        raise HTTPException(409, "Email already exists")
    finally:
        conn.close()  # <-- Don't forget to close

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/token")
@limiter.limit("10/minute")
def login(request: Request, email: str = Body(...), password: str = Body(...)):
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()
        
        if not user or not pwd.verify(password, user['password_hash']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user['id'])}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    finally:
        conn.close()

def get_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials  # this extracts the token
  #  print("TOKEN RECEIVED:", token[:20])  # first 20 chars only
  #  print("SECRET_KEY USED:", SECRET_KEY)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
      #  print("DECODED PAYLOAD:", payload)
        user_id: str = payload.get("sub")
        if user_id is None:
            print("ERROR: No sub in payload")
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError as e:
        print("JWT ERROR:", str(e))  # Signature verification failed / Expired
        raise HTTPException(status_code=401, detail="Invalid token, please login again")

# ====== MEMORY HELPERS ======

def embed_text(text: str, task="retrieval.passage"):
    headers = {
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "jina-embeddings-v3",
        "input": [text],
        "task": task # important for memory/RAG
    }
    response = requests.post("https://api.jina.ai/v1/embeddings", headers=headers, json=data)
    response.raise_for_status()
    return response.json()['data'][0]['embedding']

def save_memory(user_id, content, category="general", importance=0.5, emotion=None, privacy=False):
    try:
        content = str(content).replace("\x00", "").replace("\0", "")
        content = "".join(c for c in content if c.isprintable() or c in "\n\r\t ").strip()
        if len(content) < 5:
            return

        emb = embed_text(content, task="retrieval.passage")
        conn = psycopg2.connect(DATABASE_URL)
        try:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("INSERT INTO memories (id,user_id,content,embedding,category,importance,emotion,privacy_mode) VALUES (%s,%s,%s,%s::vector,%s,%s,%s,%s)",
                (str(uuid.uuid4()), str(user_id), content, emb, category, importance, emotion, privacy))
                conn.commit()
        finally:
            conn.close()
        # REMOVED update_summary from here completely
    except Exception as e:
        print(f"save_memory skipped: {e}")
        return

def search_memory(user_id, query, limit=5):  # 5 not 20
    print(f"SEARCH USER_ID: {user_id} query={query[:50]}")
    try:
        emb = embed_text(query, task="retrieval.query")
        print(f"EMBEDDING OK len={len(emb)}")
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT content, category FROM memories 
                WHERE user_id=%s 
                AND (
                    category = 'document' 
                    OR privacy_mode IS NULL 
                    OR privacy_mode = false
                )
                ORDER BY embedding <=> %s::vector 
                LIMIT %s
                """,
                (str(user_id), emb, limit)
            )
            rows = cur.fetchall()
            print(f"HITS TOTAL: {len(rows)}")
            docs = [r for r in rows if r.get('category') == 'document' or 'File:' in r.get('content','')[:20]]
            print(f"DOC HITS: {len(docs)}")

        conn.close()
        
        # TRIM content to 700 chars max to prevent truncation
        cleaned = []
        for r in rows:
            c = r['content']
            # Remove File: prefix to save tokens
            if c.startswith("File:"):
                c = c.split(" ", 2)[-1] if len(c.split(" ")) > 2 else c
            if len(c) > 700:
                c = c[:700] + "..."
            cleaned.append({"content": c})
        
        return cleaned
        
    except Exception as e:
        print(f"search_memory FAILED: {e}")
        import traceback; traceback.print_exc()
        return []
        
def update_summary(user_id):
    try:
        facts = search_memory(user_id, "facts goals likes", 15)
        if not facts:
            return

        # FIX: Trim to stay under Groq 8000 TPM limit
        combined = ""
        for f in facts:
            txt = str(f).replace("\x00", "")[:400]  # max 400 chars per fact
            if len(combined) + len(txt) > 3000:  # max 3000 chars total
                break
            combined += txt + "\n"

        if len(combined) < 20:
            return

        res = groq_chat(
            messages=[
                {"role": "system", "content": "Summarize into JSON {likes:[],goals:[],facts:[]}. Keep it under 200 words."},
                {"role": "user", "content": combined[:3000]}
            ],
            max_tokens=300
        ).choices[0].message.content

        # Clean NUL bytes
        res_clean = res.replace("\x00", "").strip()
        
        # Ensure it's valid JSON, if not, wrap it
        try:
            json_obj = json.loads(res_clean)
        except:
            # If model returned text not JSON, make it a fact
            json_obj = {"likes": [], "goals": [], "facts": [res_clean[:500]]}

        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO memory_hive (user_id, summary) VALUES (%s,%s::jsonb) ON CONFLICT (user_id) DO UPDATE SET summary=%s::jsonb", 
                        (str(user_id), json.dumps(json_obj), json.dumps(json_obj)))
            conn.commit()
        conn.close()
        print("Summary updated")
        
    except Exception as e:
        print(f"update_summary skipped: {e}")
        return

# ====== TIER 1-3: MEMORY + CHAT + MULTIMODAL ======
@app.get("/health")
def health():
    return {"status": "ok"}



def clean_keep_bold(txt: str):
    txt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', txt, flags=re.DOTALL)
    txt = re.sub(r'__(.*?)__', r'<b>\1</b>', txt, flags=re.DOTALL)
    txt = re.sub(r'^#{1,3}\s*(.+)$', r'<b>\1</b>', txt, flags=re.MULTILINE)
    txt = re.sub(r'`{1,3}', '', txt)
    txt = txt.replace('**', '').replace('__', '')
    return txt.strip()

def generate_image_url(prompt: str) -> str:
    encoded = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&seed={int(time.time())}"

@app.post("/generate-image")
async def generate_image(request: dict):
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "no prompt"}
    image_url = generate_image_url(prompt)
    return {"image_url": image_url, "prompt": prompt}

# Chat
last_summary_update = defaultdict(float)

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest, background_tasks: BackgroundTasks, user_id: str = Depends(get_user)):
    print(f"CHAT USER_ID: {user_id}")
    
    # === IMAGE GENERATION - MUST BE FIRST ===
    lower = req.message.lower()
    image_keywords = ["generate image", "create image", "draw ", "make image", "generate a image", "create a picture"]
    if any(k in lower for k in image_keywords):
        prompt = req.message
        # case-insensitive removal
        for kw in sorted(image_keywords, key=len, reverse=True):
            if kw in lower:
                idx = lower.find(kw)
                prompt = req.message[idx+len(kw):].strip()
                break
        if len(prompt) < 3:
            prompt = req.message
        
        image_url = generate_image_url(prompt)
        reply = f"<b>Generated:</b> {prompt}\n\nIMAGE::{image_url}"
        return {"reply": reply}
    
    raw_mems = search_memory(user_id, req.message, 5)
    print(f"DEBUG SEARCH: {len(raw_mems)} hits")

    clean = []
    for m in raw_mems:
        c = m.get('content','') if isinstance(m, dict) else getattr(m,'content','') or str(m)
        c = c.replace("\x00","").strip()
        if not c: 
            continue
        if c.startswith("File:"):
            first_space = c.find(" ")
            second_space = c.find(" ", first_space+1) if first_space != -1 else -1
            if second_space != -1:
                c = c[second_space+1:]
        if len(c) > 700:
            c = c[:700]
        clean.append(c)

    context = "\n\n---\n\n".join(clean)[:3000]
    user_msg = req.message.replace("\x00","")[:1500]
    system = f"You are {req.persona}. Roast {req.roast_level}/10. Name OBANOR. Use DOCUMENTS to answer if relevant, otherwise chat normally.\n\nDOCUMENTS:\n{context}"

    res = groq_chat(
        messages=[
            {"role":"system","content": system[:4000]},
            {"role":"user","content": user_msg}
        ], 
        max_tokens=1200
    )
    reply = res.choices[0].message.content
    reply = clean_keep_bold(reply)

    try:
        save_memory(user_id, f"User: {user_msg[:300]}\nAI: {reply[:300]}", "conversation", 0.7, privacy=True)
    except: 
        pass

    now = time.time()
    if now - last_summary_update.get(user_id,0) > 300:
        background_tasks.add_task(update_summary, user_id)
        last_summary_update[user_id] = now

    return {"reply": reply}


# Upload helpers
user_last_upload = defaultdict(float)

def save_chunks_background(user_id: str, filename: str, chunks: list, start_idx: int):
    print(f"BG SAVE USER_ID: {user_id} file={filename} start={start_idx} total={len(chunks)} privacy=True")
    for idx in range(start_idx, len(chunks)):
        try:
            clean = chunks[idx].replace("\x00", "").strip()
            if not clean:
                continue
            save_memory(user_id, f"File: {filename} {clean}", "document", 0.8, privacy=True)
            print(f"CHUNK {idx} SAVED OK privacy=True len={len(clean)}")
            time.sleep(0.2)
        except Exception as e:
            print(f"Chunk {idx} skip: {e}")
            continue

def chunk_text_groq(txt: str, chunk_size: int = 1000, overlap: int = 200):
    if len(txt) <= chunk_size:
        return [txt]
    chunks = []
    start = 0
    while start < len(txt):
        end = start + chunk_size
        if end < len(txt):
            bp = txt.rfind('\n', end-400, end)
            if bp == -1:
                bp = txt.rfind('. ', end-400, end)
            if bp == -1:
                bp = txt.rfind(' ', end-200, end)
            if bp != -1:
                end = bp + 1
        c = txt[start:end].strip()
        if c and len(c) > 50:
            chunks.append(c)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(txt):
            break
    return chunks

@app.post("/upload")
async def upload(
    file: UploadFile, 
    background_tasks: BackgroundTasks, 
    user_id: str = Depends(get_user),
    filename: str = Form(None)
):
    now = time.time()
    if now - user_last_upload.get(user_id, 0) < 10:
        return {"reply": "Please wait 10s between uploads", "status": "rate_limited"}
    user_last_upload[user_id] = now

    try:
        file_bytes = await file.read()
        original_filename = filename or file.filename or "document.pdf"
        
        if ".tmp" in original_filename or original_filename.startswith("upload"):
            ext = ".pdf" if file.content_type == "application/pdf" else ".docx" if "word" in (file.content_type or "") else ".pdf"
            original_filename = f"document{ext}"
            print(f"Fixed .tmp to: {original_filename}")

        print(f"Upload: REAL FILE: {original_filename} {len(file_bytes)} bytes USER_ID: {user_id} type={file.content_type}")

        text = ""
        try:
            reader = PdfReader(BytesIO(file_bytes))
            parts = [p.extract_text() or "" for p in reader.pages]
            text = "\n".join(parts)
        except:
            try:
                text = "\n".join([p.text for p in Document(BytesIO(file_bytes)).paragraphs])
            except:
                text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return {"reply": f"Could not read file: {e}"}

    text = text.replace("\x00", "").strip()
    text = "".join(c for c in text if c.isprintable() or c in "\n\r\t ").strip()
    if not text or len(text) < 20:
        return {"reply": "File is empty or scanned image PDF"}

    chunks = chunk_text_groq(text, 1000, 200)
    print(f"EXTRACTED OK: {len(text)} chars -> {len(chunks)} chunks File: {original_filename}")

    for idx in range(min(2, len(chunks))):
        try:
            save_memory(user_id, f"File: {original_filename} {chunks[idx]}", "document", 0.8, privacy=True)
            print(f"CHUNK {idx} SAVED OK privacy=True File: {original_filename}")
        except Exception as e:
            print(f"chunk {idx} failed: {e}")

    if len(chunks) > 2:
        background_tasks.add_task(save_chunks_background, user_id, original_filename, chunks, 2)

    return {
        "reply": f"File {original_filename} saved: {len(chunks)} chunks (private). Ask me anything. (Note:Be specific about the file name, not to confuse the AI, so you don't get wrong reply). File will fully save for 4 to 30 seconds depending on how large the file is, then the AI can have access to the full content of the file, first two pages saves for two seconds so the AI can reply correctly regarding the first two pages in the first two seconds.",
        "status": "indexed",
        "chunks": len(chunks)
    }



@app.post("/summarize_full")
@limiter.limit("10/minute")
async def summarize_full(request: Request, file: UploadFile, user_id: str = Depends(get_user)):
    """EXPENSIVE - Burns 5k-15k tokens. Use only when user clicks 'Full Summary'"""
    text = ""
    try:
        if file.filename.lower().endswith(".pdf"):
            text = "".join([p.extract_text() or "" for p in PdfReader(file.file).pages])
        elif file.filename.lower().endswith(".docx"):
            text = "\n".join([p.text for p in Document(file.file).paragraphs])
        else:
            text = (await file.read()).decode("utf-8", errors="ignore")
    except Exception as e:
        return {"reply": f"Could not read file: {e}"}

    text = text.replace("\x00", "")[:20000]  # cap at 20k chars
    if not text.strip():
        return {"reply": "File empty"}

    try:
        res = groq_chat(
            messages=[
                {"role": "system", "content": "You are OBANOR. Summarize entire document with key facts, dates, names."},
                {"role": "user", "content": f"Summarize full document {file.filename}:\n\n{text}"}
            ],
            max_tokens=500
        )
        return {"reply": res.choices[0].message.content}
    except Exception as e:
        if "429" in str(e):
            return {"reply": "Daily limit hit (200k). Your file is already saved via /upload and searchable. Try full summary tomorrow or upgrade."}
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vision")
async def vision(image: UploadFile, user_id: str = Depends(get_user)):
    # upload to cloudinary to get URL
    upload_res = cloudinary.uploader.upload(image.file)
    image_url = upload_res['secure_url']

    # Use REAL vision model
    res = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{
            "role":"user",
            "content": [
                {"type":"text","text":"Describe this image in detail and answer what the user might want to know."},
                {"type":"image_url","image_url":{"url": image_url}}
            ]
        }]
    )
    description = res.choices[0].message.content
    save_memory(user_id, f"Image: {description}", "vision")
    return {"reply": description, "description": description, "image_url": image_url}@app.post("/vision")
async def vision(image: UploadFile, user_id: str = Depends(get_user)):
    # upload to cloudinary to get URL
    upload_res = cloudinary.uploader.upload(image.file)
    image_url = upload_res['secure_url']

    # Use REAL vision model
    res = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{
            "role":"user",
            "content": [
                {"type":"text","text":"Describe this image in detail and answer what the user might want to know."},
                {"type":"image_url","image_url":{"url": image_url}}
            ]
        }]
    )
    description = res.choices[0].message.content
    save_memory(user_id, f"Image: {description}", "vision")
    return {"reply": description, "description": description, "image_url": image_url}

@app.post("/voice-to-text")
async def stt(audio: UploadFile = File(...), file: UploadFile = File(None)):
    # Accept both 'audio' and 'file' field names from Android
    actual_file = file or audio
    try:
        transcription = client.audio.transcriptions.create(
            file=(actual_file.filename, await actual_file.read()),
            model="whisper-large-v3-turbo"
        )
        return {"text": transcription.text, "reply": transcription.text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Transcription failed: {e}")

@app.post("/tts")
def tts(text: str = Body(..., embed=True)):
    try:
        speech = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="Fritz-PlayAI",
            input=text,
            response_format="wav"
        )
        # Return audio file directly so app can play it
        return StreamingResponse(
            io.BytesIO(speech.read() if hasattr(speech, 'read') else speech),
            media_type="audio/wav"
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ====== TIER 4-5: PRODUCTIVITY + AGENT ======
@app.post("/task")
def add_task(title: str, due_at: str, user_id: str = Depends(get_user)):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO tasks (id,user_id,title,due_at,done) VALUES (%s,%s,%s,%s,false)", (str(uuid.uuid4()), str(user_id), title, due_at));
    conn.commit()
    conn.close()
    return {"status":"task added"}

@app.post("/calendar/create") #28
def create_event(title: str, start: str, user_id: str = Depends(get_user)):
    creds = Credentials(token=os.getenv("GOOGLE_TOKEN"))
    service = build('calendar', 'v3', credentials=creds)
    event = service.events().insert(calendarId='primary', body={"summary": title, "start": {"dateTime": start}}).execute()
    return event

@app.post("/email/summarize") #29
def email_summary(user_id: str = Depends(get_user)):
    # Wire Gmail API here
    return {"summary": "3 unread. 1 from Tolu about deadline"}

@app.post("/search") #33
def web_search(query: str):
    return requests.post("https://api.tavily.com/search", json={"api_key":os.getenv("TAVILY_KEY"),"query":query}).json()

#@app.post("/browser/book") #37
#def browser_book(url: str, action: str):
   # with sync_playwright() as p:
      #  browser = p.chromium.launch(headless=True)
       # page = browser.new_page()
       # page.goto(url)
        # page.click(action)
       # browser.close()
   # return {"status":"done"}

# ====== TIER 6-7-8: SOCIAL + DEV + SAFETY ======
@app.post("/contact")
def add_contact(name: str, notes: str, user_id: str = Depends(get_user)):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO contacts (id,user_id,name,notes) VALUES (%s,%s,%s,%s)", (str(uuid.uuid4()), str(user_id), name, notes));
    conn.commit()  
    conn.close()
    return {"status": "done"}

@app.get("/analytics")
def analytics(user_id: str = Depends(get_user)):
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT category, COUNT(*) FROM memories WHERE user_id=%s GROUP BY category", (str(user_id,)));
        result = cur.fetchall()
    conn.close()
    return {"topics": result}

@app.get("/export")
def export(user_id: str = Depends(get_user)):
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM memories WHERE user_id=%s", (str(user_id),));
        data = cur.fetchall()
    conn.close()
    return {"data": data}

@app.post("/webhook") #48
def add_webhook(trigger: str, action: str, user_id: str = Depends(get_user)):
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO webhooks (id,user_id,trigger,action) VALUES (%s,%s,%s,%s)", (str(uuid.uuid4()), str(user_id), trigger, action));
    conn.commit()
    conn.close()
    return {"status": "webhook saved"}

@app.post("/backup") #54
def backup():
    # pg_dump -> upload to cloudinary
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO backups (id,created_at,url) VALUES (%s,%s,%s)", (str(uuid.uuid4()), datetime.now(), "backup_url"));
    conn.commit()
    conn.close()
    return {"status":"backed up"}

# ====== TIER 4: PROACTIVE #22 ======
def proactive_check():
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT user_id,title FROM tasks WHERE due_at < %s AND done=false", (datetime.now() + timedelta(hours=1),))
        rows = cur.fetchall()
    conn.close()
    for user_id, title in rows:
        print(f"Remind {user_id}: {title} due soon") # send whatsapp here

scheduler.add_job(proactive_check, 'interval', minutes=30)
scheduler.start()