import io
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from deepface import DeepFace

app = FastAPI(title="MedAssist DeepFace API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path("data")
DATA_PATH.mkdir(exist_ok=True)
DB_FILE = DATA_PATH / "faces.json"

# emergency_id -> {"embedding": list[float], "profile": {...}}
if DB_FILE.exists():
    with open(DB_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
else:
    registry = {}

# ------------------- models -------------------
class Profile(BaseModel):
    name: Optional[str] = None
    blood_group: Optional[str] = None
    allergies: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    medications: Optional[List[str]] = None

# ------------------- helpers ------------------

def _img_to_embedding(image_bytes: bytes) -> List[float]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(pil_img)
    try:
        rep = DeepFace.represent(img, model_name="Facenet512", detector_backend="opencv", enforce_detection=True)
        return rep[0]["embedding"]  # deepface returns list[dict]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def _save_db():
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f)

# ------------------- routes -------------------

@app.post("/face/register")
async def register_face(
    emergency_id: str = Form(...),
    image: UploadFile = File(...),
    profile_json: Optional[str] = Form(None),
):
    if not emergency_id:
        raise HTTPException(status_code=400, detail="emergency_id required")
    image_bytes = await image.read()
    embedding = _img_to_embedding(image_bytes)
    profile = json.loads(profile_json) if profile_json else {}

    registry[emergency_id] = {"embedding": embedding, "profile": profile}
    _save_db()
    return {"status": "registered", "emergency_id": emergency_id}

@app.post("/face/identify")
async def identify_face(image: UploadFile = File(...)):
    if not registry:
        raise HTTPException(status_code=404, detail="No registered faces")
    image_bytes = await image.read()
    emb = _img_to_embedding(image_bytes)

    # simple cosine similarity
    def cosine(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    best_id = None
    best_score = -1.0
    for eid, data in registry.items():
        score = cosine(emb, data["embedding"])
        if score > best_score:
            best_score = score
            best_id = eid

    if best_score > 0.4:  # threshold tweak
        return {
            "match": True,
            "similarity": float(best_score),
            "emergency_id": best_id,
            "profile": registry[best_id]["profile"],
        }
    return {"match": False, "similarity": float(best_score)}

@app.get("/")
async def root():
    return {"status": "OK", "registered": len(registry)}
