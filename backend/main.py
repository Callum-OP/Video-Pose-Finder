from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from processor import process_frames

app = FastAPI()

# Allow React to talk to this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "https://callum-op.github.io/Video-Pose-Finder/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data models matching current JSON format ────────────────────────────
class Landmark(BaseModel):
    x: float
    y: float
    z: float
    v: float # Visibility

class Frame(BaseModel):
    frameIndex: int
    timeMs: float
    landmarks: List[Landmark]
    worldLandmarks: Optional[List[Landmark]] = None

class EnhanceRequest(BaseModel):
    frames: List[Frame]

class EnhanceResponse(BaseModel):
    frames: List[Frame]
    message: str

# ── Health check  ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is running"}

# ── Main enhance endpoint ─────────────────────────────────────────────────────
@app.post("/enhance")
def enhance(req: EnhanceRequest):
    improved = process_frames(req.frames)
    return {"frames": improved, "message": f"Enhanced {len(improved)} frames"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)