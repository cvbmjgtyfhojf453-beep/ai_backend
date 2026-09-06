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
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer
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
from jose import jwt
from passlib.context import CryptContext #52 Auth
import requests
from apscheduler.schedulers.background import BackgroundScheduler #22 Proactive
from google.oauth2.credentials import Credentials #28 Calendar
from googleapiclient.discovery import build
# from playwright.sync_api import sync_playwright #37 Browser
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI OS Backend - Tier 1-8")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
oauth2 = OAuth2PasswordBearer(tokenUrl="token")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
scheduler = BackgroundScheduler()

# DB + VECTOR - Auto create tables
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DATABASE_URL = os.getenv("DATABASE_URL")

def create_tables():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
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
CREATE TABLE users2 (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    full_name TEXT,
    avatar_url TEXT,
    bio TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- TIER 2: FRIENDS + BLOCKS
CREATE TABLE friendships (
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
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- TIER 4: COMMENTS + LIKES
CREATE TABLE comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id, user_id)
);

-- TIER 5: CHATS + MESSAGES
CREATE TABLE chats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    is_group BOOLEAN DEFAULT FALSE,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_participants (
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chats(id) ON DELETE CASCADE,
    sender_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- TIER 6-8: INDEXES FOR PERFORMANCE
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX idx_comments_post_id ON comments(post_id);
CREATE INDEX idx_likes_post_id ON likes(post_id);
CREATE INDEX idx_messages_chat_id ON messages(chat_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX idx_friendships_user_id ON friendships(user_id);
CREATE INDEX idx_friendships_friend_id ON friendships(friend_id);
CREATE INDEX idx_friendships_status ON friendships(status);

-- TIER 7: AUTO UPDATE updated_at TRIGGER
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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
def register(request: Request, email: str, password: str):
    hash = pwd.hash(password)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (id,email,password_hash) VALUES (%s,%s,%s)", (uuid.uuid4(), email, hash))
            conn.commit()
        return {"status":"user created"}
    except:
        conn.rollback()
        raise HTTPException(409, "Email already exists")

@app.post("/token")
@limiter.limit("10/minute")
def login(request: Request, email: str, password: str):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id,password_hash FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
    if not user or not pwd.verify(password, user['password_hash']):
        raise HTTPException(401, "Invalid credentials")
    token = jwt.encode({"sub": str(user['id']), "exp": datetime.utcnow() + timedelta(days=7)}, os.getenv("SECRET_KEY"), algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}

def get_user(token: str = Depends(oauth2)):
    try:
        payload = jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        return payload["sub"]
    except:
        raise HTTPException(401, "Invalid token")

# ====== MEMORY HELPERS ======
def embed_text(text: str):
    return client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

def save_memory(user_id, content, category="general", importance=0.5, emotion=None, privacy=False):
    emb = embed_text(content)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO memories (id,user_id,content,embedding,category,importance,emotion,privacy_mode) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (uuid.uuid4(), user_id, content, emb, category, importance, emotion, privacy));
        conn.commit()
    if not privacy: update_summary(user_id)

def search_memory(user_id, query, limit=5):
    emb = embed_text(query)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT content FROM memories WHERE user_id=%s AND privacy_mode=false ORDER BY embedding <=> %s LIMIT %s", (user_id, emb, limit));
        return [r['content'] for r in cur.fetchall()]

def update_summary(user_id):
    facts = search_memory(user_id, "facts goals likes", 20)
    res = client.chat.completions.create(model="llama-3.1-8b", messages=[
        {"role":"system","content":"Summarize into JSON {likes:[],goals:[],facts:[]}"},
        {"role":"user","content":str(facts)}
    ]).choices[0].message.content
    with conn.cursor() as cur:
        cur.execute("INSERT INTO memory_hive (user_id, summary) VALUES (%s,%s) ON CONFLICT (user_id) DO UPDATE SET summary=%s", (user_id, res, res));
        conn.commit()

# ====== TIER 1-3: MEMORY + CHAT + MULTIMODAL ======
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, message: str, persona: str="assistant", roast_level: int=0, user_id: str = Depends(get_user)):
    context = "\n".join(search_memory(user_id, message))
    system = f"You are {persona}. Roast level {roast_level}/10. Facts: {context}. Reply in user's language: Yoruba, Pidgin, English."
    res = client.chat.completions.create(model="llama-3.1-8b", messages=[{"role":"system","content":system},{"role":"user","content":message}])
    reply = res.choices[0].message.content
    save_memory(user_id, f"User: {message}\nAI: {reply}")
    return {"reply": reply}

@app.post("/upload")
async def upload(file: UploadFile, user_id: str = Depends(get_user)):
    text = ""
    if file.filename.endswith(".pdf"):
        text = "".join([p.extract_text() for p in PdfReader(file.file).pages])
    elif file.filename.endswith(".docx"):
        text = "\n".join([p.text for p in Document(file.file).paragraphs])
    for chunk in [text[i:i+500] for i in range(0, len(text), 500)]:
        save_memory(user_id, chunk, "document", 0.8)
    return {"status":"indexed"}

@app.post("/vision")
async def vision(image: UploadFile, user_id: str = Depends(get_user)):
    upload_res = cloudinary.uploader.upload(image.file)
    res = client.chat.completions.create(model="llama-3.2-11b-vision", messages=[{"role":"user","content":[{"type":"text","text":"Describe this image"},{"type":"image_url","image_url":{"url":upload_res['url']}}]}])
    save_memory(user_id, f"Image: {res.choices[0].message.content}", "vision")
    return {"description": res.choices[0].message.content}

@app.post("/voice-to-text")
async def stt(audio: UploadFile):
    return {"text": client.audio.transcriptions.create(file=(audio.filename, audio.file.read()), model="whisper-large-v3").text}

@app.post("/tts")
def tts(text: str):
    return {"audio": client.audio.speech.create(model="playai-tts", voice="Fritz-PlayAI", input=text).content}

# ====== TIER 4-5: PRODUCTIVITY + AGENT ======
@app.post("/task")
def add_task(title: str, due_at: str, user_id: str = Depends(get_user)):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO tasks (id,user_id,title,due_at,done) VALUES (%s,%s,%s,%s,false)", (uuid.uuid4(), user_id, title, due_at));
        conn.commit()
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
    with conn.cursor() as cur:
        cur.execute("INSERT INTO contacts (id,user_id,name,notes) VALUES (%s,%s,%s,%s)", (uuid.uuid4(), user_id, name, notes));
        conn.commit()

@app.get("/analytics")
def analytics(user_id: str = Depends(get_user)):
    with conn.cursor() as cur:
        cur.execute("SELECT category, COUNT(*) FROM memories WHERE user_id=%s GROUP BY category", (user_id,));
        return {"topics": cur.fetchall()}

@app.get("/export")
def export(user_id: str = Depends(get_user)):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM memories WHERE user_id=%s", (user_id,));
        return {"data": cur.fetchall()}

@app.post("/webhook") #48
def add_webhook(trigger: str, action: str, user_id: str = Depends(get_user)):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO webhooks (id,user_id,trigger,action) VALUES (%s,%s,%s,%s)", (uuid.uuid4(), user_id, trigger, action));
        conn.commit()

@app.post("/backup") #54
def backup():
    # pg_dump -> upload to cloudinary
    with conn.cursor() as cur:
        cur.execute("INSERT INTO backups (id,created_at,url) VALUES (%s,%s,%s)", (uuid.uuid4(), datetime.now(), "backup_url"));
        conn.commit()
    return {"status":"backed up"}

# ====== TIER 4: PROACTIVE #22 ======
def proactive_check():
    with conn.cursor() as cur:
        cur.execute("SELECT user_id,title FROM tasks WHERE due_at < %s AND done=false", (datetime.now() + timedelta(hours=1),))
        for user_id, title in cur.fetchall():
            print(f"Remind {user_id}: {title} due soon") # send whatsapp here

scheduler.add_job(proactive_check, 'interval', minutes=30)
scheduler.start()