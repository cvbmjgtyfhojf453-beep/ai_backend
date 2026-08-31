from fastapi.responses import JSONResponse
from typing import List, Dict
import traceback
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Request
from pydantic import BaseModel
import bcrypt
import jwt
import httpx
import os, json, uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request, BackgroundTasks, Header
from fastapi.security import OAuth2PasswordBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from groq import Groq
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader
from docx import Document
from jose import jwt
from passlib.context import CryptContext #52 Auth
import requests
from apscheduler.schedulers.background import BackgroundScheduler #22 Proactive
from google.oauth2.credentials import Credentials #28 Calendar
from googleapiclient.discovery import build
#from playwright.sync_api import sync_playwright #37 Browser
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI OS Backend - Tier 1-8")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

JWT_SECRET = os.getenv("JWT_SECRET", "supersecret")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
oauth2 = OAuth2PasswordBearer(tokenUrl="token")
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
scheduler = BackgroundScheduler()

# DB + VECTOR
conn = psycopg2.connect(os.getenv("DATABASE_URL"), sslmode='require')
register_vector(conn)
cur = conn.cursor()

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

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.replace("Bearer ", "")
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    return payload["sub"]

@app.post("/chat")
def chat(message: dict, user_id: str = Depends(get_current_user)):
    # For now just echo back + save to memory
    return {"user_id": user_id, "reply": f"You said: {message['message']}"}

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
 #   with sync_playwright() as p:
 #       browser = p.chromium.launch(headless=True)
 #      page = browser.new_page()
 #       page.goto(url)
 #       # page.click(action)
 #       browser.close()
 #   return {"status":"done"}

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




class UserCreate(BaseModel):
    email: str
    password: str

# FAKE DB for now - later we connect to Postgres
users_db = {}

@app.post("/auth/register")
def register(user: UserCreate):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())
    user_id = str(uuid.uuid4())
    users_db[user.email] = {"id": user_id, "email": user.email, "password": hashed}
    
    token = jwt.encode({"sub": user_id, "exp": datetime.utcnow() + timedelta(days=7)}, JWT_SECRET, algorithm="HS256")
    return {"access_token": token}

@app.post("/auth/login")
def login(user: UserCreate):
    db_user = users_db.get(user.email)
    if not db_user or not bcrypt.checkpw(user.password.encode(), db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode({"sub": db_user["id"], "exp": datetime.utcnow() + timedelta(days=7)}, JWT_SECRET, algorithm="HS256")
    return {"access_token": token}