import io
import json
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
from deepface import DeepFace
from pymongo import MongoClient
from cryptography.fernet import Fernet, InvalidToken

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

MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME", "med_assist")
MONGO_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "face_data")

FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "Facenet512")
FACE_DETECTOR_BACKEND = os.getenv("FACE_DETECTOR_BACKEND", "opencv")
SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.5"))

# emergency_id -> {"embedding": list[float], "profile": {...}}
if DB_FILE.exists():
    with open(DB_FILE, "r", encoding="utf-8") as f:
        registry = json.load(f)
else:
    registry = {}

for eid, data in list(registry.items()):
    if "embedding" in data and "embeddings" not in data:
        emb = data["embedding"]
        data["embeddings"] = [emb]

mongo_client: Optional[MongoClient] = None
mongo_collection = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client[MONGO_DB_NAME]
        mongo_collection = mongo_db[MONGO_COLLECTION_NAME]
    except Exception:
        mongo_client = None
        mongo_collection = None

_FERNET: Optional[Fernet] = None


def _get_fernet() -> Optional[Fernet]:
    global _FERNET
    if _FERNET is not None:
        return _FERNET
    key = os.getenv("FACE_ENCRYPTION_KEY")
    if not key:
        _FERNET = None
        return None
    try:
        _FERNET = Fernet(key)
    except Exception:
        _FERNET = None
    return _FERNET

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
        rep = DeepFace.represent(
            img,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_DETECTOR_BACKEND,
            enforce_detection=True,
        )
        return rep[0]["embedding"]  # deepface returns list[dict]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def _save_db():
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f)


def _store_face_record(emergency_id: str, embedding: List[float], profile: dict) -> None:
    f = _get_fernet()
    emb_to_store: object
    if f is not None:
        payload = json.dumps(embedding).encode("utf-8")
        emb_to_store = f.encrypt(payload).decode("utf-8")
    else:
        emb_to_store = embedding
    if mongo_collection is not None:
        now = datetime.utcnow().isoformat()
        update_doc = {
            "$setOnInsert": {"created_at": now},
            "$set": {"profile": profile or {}, "updated_at": now},
            "$push": {"embeddings": emb_to_store},
        }
        mongo_collection.update_one({"emergency_id": emergency_id}, update_doc, upsert=True)
    else:
        data = registry.get(emergency_id, {})
        embeddings = data.get("embeddings") or []
        embeddings.append(emb_to_store)
        data["embeddings"] = embeddings
        if profile:
            data["profile"] = profile
        elif "profile" not in data:
            data["profile"] = {}
        registry[emergency_id] = data
        _save_db()


def _iter_face_records():
    def _decode_embeddings(raw_list):
        f = _get_fernet()
        decoded: List[List[float]] = []
        for item in raw_list or []:
            if isinstance(item, list):
                decoded.append(item)
            elif isinstance(item, str) and f is not None:
                try:
                    data = f.decrypt(item.encode("utf-8"))
                    decoded.append(json.loads(data.decode("utf-8")))
                except (InvalidToken, json.JSONDecodeError):
                    continue
        return decoded

    if mongo_collection is not None:
        for doc in mongo_collection.find({}):
            yield {
                "emergency_id": doc.get("emergency_id"),
                "embeddings": _decode_embeddings(doc.get("embeddings") or []),
                "profile": doc.get("profile") or {},
            }
    else:
        for eid, data in registry.items():
            yield {
                "emergency_id": eid,
                "embeddings": _decode_embeddings(data.get("embeddings") or []),
                "profile": data.get("profile") or {},
            }


def _count_registered() -> int:
    if mongo_collection is not None:
        return mongo_collection.count_documents({})
    return len(registry)

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

    _store_face_record(emergency_id, embedding, profile)
    return {"status": "registered", "emergency_id": emergency_id}

@app.post("/face/identify")
async def identify_face(image: UploadFile = File(...)):
    records = list(_iter_face_records())
    if not records:
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
    best_profile = None
    for record in records:
        embeddings = record["embeddings"]
        if not embeddings:
            continue
        emb_array = np.array(embeddings)
        avg_emb = emb_array.mean(axis=0)
        score = cosine(emb, avg_emb)
        if score > best_score:
            best_score = score
            best_id = record["emergency_id"]
            best_profile = record["profile"]

    if best_id is None:
        return {"match": False, "similarity": float(best_score)}

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "match": True,
            "similarity": float(best_score),
            "emergency_id": best_id,
            "profile": best_profile,
        }
    return {"match": False, "similarity": float(best_score)}

@app.get("/")
async def root():
    return {"status": "OK", "registered": _count_registered()}
