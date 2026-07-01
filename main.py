from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv() # loads.env locally. On Render use Environment Variables

app = FastAPI()

# Let Flutter call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # change to your app domain later
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Key is loaded ONLY on the server. Flutter can't access this.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # fastest + best free Groq model
        messages=[
            {"role": "system", "content": "You are a helpful AI. Be brief, warm, and direct."},
            {"role": "user", "content": req.message}
        ],
        max_tokens=512,
        temperature=0.7,
    )
    return {"reply": completion.choices[0].message.content}

@app.get("/")
def health():
    return {"status": "ok"} # used for UptimeRobot to prevent cold starts