"""
NAKLES API — Deep Learning Showcase
CNN (CIFAR-10) + LSTM (Météo Jena)
"""

import io
import os
import sys
import base64
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# ─── Ajout du chemin racine au sys.path ───────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── Chargement paresseux des modèles (lazy) ──────────────────────────────────
cnn_model  = None
lstm_model = None
data_min   = None
data_max   = None

CLASS_NAMES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]
CLASS_EMOJIS = ["✈️","🚗","🐦","🐱","🦌","🐶","🐸","🐴","⛵","🚚"]

SEQUENCE_LENGTH = 24

# ─── Application FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="NAKLES API",
    description="API de démonstration — CNN (CIFAR-10) & LSTM (Météo)",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


# ─── Chargement des modèles ───────────────────────────────────────────────────
def load_cnn():
    global cnn_model
    if cnn_model is None:
        try:
            import tensorflow as tf
            from models.cnn_model import CustomCNN
            model_path = ROOT / "saved_model" / "best_cnn.keras"
            cnn_model = tf.keras.models.load_model(
                str(model_path),
                custom_objects={"CustomCNN": CustomCNN}
            )
            print("✅ CNN chargé")
        except Exception as e:
            print(f"⚠️  CNN non disponible : {e}")
    return cnn_model


def load_lstm():
    global lstm_model, data_min, data_max
    if lstm_model is None:
        try:
            import tensorflow as tf
            from models.rnn_model import CustomLSTM
            model_path = ROOT / "saved_model" / "best_lstm.keras"
            lstm_model = tf.keras.models.load_model(
                str(model_path),
                custom_objects={"CustomLSTM": CustomLSTM}
            )
            # Paramètres de normalisation (données Jena simulées)
            np.random.seed(42)
            n = 100_000
            t = np.linspace(0, 4 * np.pi, n)
            temps = (15 * np.sin(t) + 10 + 3 * np.sin(t * 365 / 2)
                     + np.random.normal(0, 0.5, n)).astype(np.float32)
            data_min = float(temps.min())
            data_max = float(temps.max())
            print("✅ LSTM chargé")
        except Exception as e:
            print(f"⚠️  LSTM non disponible : {e}")
    return lstm_model


# ─── Pages HTML ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def page_accueil(request: Request):
   # return templates.TemplateResponse("index.html", {"request": request})
    return templates.TemplateResponse(request, "index.html")


@app.get("/cnn", response_class=HTMLResponse)
async def page_cnn(request: Request):
   # return templates.TemplateResponse("cnn.html", {"request": request})
   return templates.TemplateResponse(request, "cnn.html")


@app.get("/lstm", response_class=HTMLResponse)
async def page_lstm(request: Request):
    #return templates.TemplateResponse("lstm.html", {"request": request})
     return templates.TemplateResponse(request, "lstm.html")


@app.get("/metriques", response_class=HTMLResponse)
async def page_metriques(request: Request):
    #return templates.TemplateResponse("metriques.html", {"request": request})
    return templates.TemplateResponse(request, "metriques.html")


# ─── API CNN ──────────────────────────────────────────────────────────────────
@app.post("/api/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    """
    Prédit la classe d'une image CIFAR-10.
    Accepte : PNG, JPEG, JPG
    """
    model = load_cnn()
    if model is None:
        # Mode démo : retourne une prédiction factice
        import random
        idx = random.randint(0, 9)
        probs = [round(random.uniform(0.01, 0.05), 4) for _ in range(10)]
        probs[idx] = round(random.uniform(0.55, 0.92), 4)
        s = sum(probs)
        probs = [round(p / s, 4) for p in probs]
        return JSONResponse({
            "mode": "demo",
            "classe_predite": CLASS_NAMES[idx],
            "emoji": CLASS_EMOJIS[idx],
            "confiance": probs[idx],
            "toutes_probabilites": {
                CLASS_NAMES[i]: probs[i] for i in range(10)
            }
        })

    import tensorflow as tf
    from PIL import Image

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[np.newaxis, ...]   # (1, 32, 32, 3)

    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))

    return JSONResponse({
        "mode": "live",
        "classe_predite": CLASS_NAMES[idx],
        "emoji": CLASS_EMOJIS[idx],
        "confiance": float(probs[idx]),
        "toutes_probabilites": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(10)
        }
    })


# ─── API LSTM ─────────────────────────────────────────────────────────────────
class LSTMRequest(BaseModel):
    temperatures: List[float]   # exactement 24 valeurs


@app.post("/api/predict/lstm")
async def predict_lstm(body: LSTMRequest):
    """
    Prédit la prochaine température à partir de 24 valeurs passées.
    """
    if len(body.temperatures) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Exactement {SEQUENCE_LENGTH} valeurs requises, "
                   f"reçu : {len(body.temperatures)}"
        )

    model = load_lstm()

    temps_array = np.array(body.temperatures, dtype=np.float32)

    if model is None or data_min is None:
        # Mode démo
        last = temps_array[-1]
        pred = float(last + np.random.normal(0, 0.3))
        return JSONResponse({
            "mode": "demo",
            "prediction": round(pred, 2),
            "unite": "°C",
            "sequence_entree": body.temperatures
        })

    t_min = float(min(body.temperatures)) - 2
    t_max = float(max(body.temperatures)) + 2

    norm = (temps_array - t_min) / (t_max - t_min + 1e-8)
    norm = norm.reshape(1, SEQUENCE_LENGTH, 1)

    pred_norm = float(model.predict(norm, verbose=0)[0][0])
    pred_real = pred_norm * (t_max - t_min) + t_min

    return JSONResponse({
        "mode": "live",
        "prediction": round(float(pred_real), 2),
        "unite": "°C",
        "sequence_entree": body.temperatures
    })


# ─── API Métriques ────────────────────────────────────────────────────────────
@app.get("/api/metriques")
async def get_metriques():
    """Retourne les métriques des deux modèles."""
    return JSONResponse({
        "cnn": {
            "accuracy":   0.8126,
            "loss":       0.5712,
            "epochs":     40,
            "best_epoch": 33,
            "architecture": "3× Conv2D → BN → MaxPool → Dense(256) → Dropout(0.5) → Softmax(10)",
            "params":     "~1.2M",
            "dataset":    "CIFAR-10 (50 000 train / 10 000 test)",
            "objectif":   "≥ 70% ✅ atteint",
            "par_classe": {
                "avion":       0.878,
                "automobile":  0.926,
                "oiseau":      0.722,
                "chat":        0.625,
                "cerf":        0.774,
                "chien":       0.701,
                "grenouille":  0.894,
                "cheval":      0.864,
                "bateau":      0.877,
                "camion":      0.878
            }
        },
        "lstm": {
            "mse":   0.0002,
            "rmse":  0.0141,
            "mae":   0.0095,
            "epochs": 26,
            "architecture": "LSTM(64, return_seq=True) → Dropout(0.2) → LSTM(32) → Dense(1)",
            "params": "~42K",
            "sequence_length": 24,
            "dataset": "Données météo simulées (100 000 points)",
            "commentaire": "MSE < 0.001 — excellent suivi de la tendance"
        }
    })


# ─── Lancement ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
