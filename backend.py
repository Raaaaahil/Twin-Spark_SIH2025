# backend.py
import os
import sys
import shutil
import subprocess
import re
import json
from pathlib import Path
from typing import Optional

import torch  # make sure torch is installed in your venv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# CONFIG - change if your checkpoints or classes are in other paths
CHECKPOINT_PATH = "checkpoints/epoch_30.pth"   # default checkpoint to use
CLASSES_FILE = "data/classes.txt"
PREDICTIONS_CSV = "results/predictions.csv"
IMG_SIZE = "224"
TOPK_DEFAULT = "3"

app = FastAPI(title="Intelligent Sprinkler - Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# simple homepage with upload form (open this in browser)
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        """
        <html>
        <head><title>Intelligent Sprinkler - Upload</title></head>
        <body>
          <h2>Upload leaf image for prediction</h2>
          <form action="/predict" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*"/>
            <input type="submit" value="Upload & Predict"/>
          </form>
          <p>Or use /docs for Swagger UI.</p>
        </body>
        </html>
        """
    )

def get_training_accuracy_from_checkpoint(path: str) -> Optional[float]:
    """
    Try to read training/eval accuracy values from checkpoint metadata.
    Many checkpoints store dicts like {'epoch':N,'model_state_dict':..., 'accuracy': X}
    """
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if isinstance(ckpt, dict):
        # try common keys
        for key in ("best_acc", "accuracy", "val_acc", "acc", "best_accuracy"):
            if key in ckpt:
                try:
                    return float(ckpt[key])
                except Exception:
                    pass
        # sometimes nested under metadata
        for k, v in ckpt.items():
            if isinstance(v, (int, float)) and 0 <= float(v) <= 100:
                # rough heuristic but avoid false positives
                pass
    return None

def run_predict_subprocess(image_path: str, checkpoint: str = CHECKPOINT_PATH, topk: str = TOPK_DEFAULT):
    """
    Run your existing CLI prediction module and capture stdout/stderr.
    It calls: python -m src.predict_and_infect --image_path <image_path> --checkpoint <checkpoint> ...
    Returns dict with stdout, stderr, returncode.
    """
    python_exe = sys.executable  # ensures same venv python is used
    cmd = [
    python_exe,
    "-m",
    "src.predict_and_infect",
    "--image_path", image_path,
    "--checkpoint", checkpoint,
    "--classes_file", CLASSES_FILE,
    "--img_size", IMG_SIZE,
    "--topk", topk
]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"stdout": proc.stdout, "stderr": proc.stderr, "rc": proc.returncode}

def parse_stdout_for_metrics(stdout: str):
    """
    Parse the CLI output to extract:
      - model_accuracy (the script prints something like 'Model accuracy from training/evaluation: 57.47%')
      - top1_confidence (line like 'Top-1 confidence: 99.60%')
      - infection_percentage (line like 'Infection percentage: 13.57%')
    Returns dict with floats (or None if not found).
    """
    def find_pct(label):
        m = re.search(rf"{label}[: ]+\s*([0-9]+(?:\.[0-9]+)?)\s*%", stdout, re.IGNORECASE)
        return float(m.group(1)) if m else None

    model_acc = find_pct(r"Model accuracy(?: from training/evaluation)?")
    top1 = find_pct(r"Top-1 confidence")
    infect = find_pct(r"Infection percentage")
    # also try other phrasings
    if model_acc is None:
        m = re.search(r"Model accuracy from training:.*?([0-9]+(?:\.[0-9]+)?)\s*%", stdout, re.IGNORECASE)
        if m: model_acc = float(m.group(1))
    return {"model_accuracy": model_acc, "top1_confidence": top1, "infection_percentage": infect}

def recommend_action(infect_pct: float) -> str:
    """
    Convert infection percentage to a short recommended action level.
    Thresholds (example):
      - >= 50% : Immediate / High
      - 20-50% : Medium — treat soon
      - 5-20%  : Low — monitor and consider localized spray
      - <5%    : None — monitor
    """
    if infect_pct is None:
        return "No recommendation (infection percent unknown)."
    if infect_pct >= 50:
        return "High: Immediate treatment recommended (spray / isolate)."
    if infect_pct >= 20:
        return "Medium: Treat soon — targeted spraying suggested."
    if infect_pct >= 5:
        return "Low: No immediate full-spraying; monitor and consider spot treatment."
    return "None: No treatment needed. Monitor the plant."

@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    # ensure uploads dir exists
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    # save file to disk
    try:
        ext = Path(file.filename).suffix or ".jpg"
        out_path = uploads_dir / f"upload_{os.getpid()}{ext}"
        with out_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # attempt to get training accuracy from checkpoint metadata (if present)
    training_acc = get_training_accuracy_from_checkpoint(CHECKPOINT_PATH)
    # run CLI predictor
    result = run_predict_subprocess(str(out_path), checkpoint=CHECKPOINT_PATH, topk=str(topk))

    # if subprocess failed, return stderr for debugging
    if result["rc"] != 0:
        return JSONResponse(status_code=500, content={
            "error": "Internal prediction error",
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "cmd_rc": result["rc"]
        })

    # parse stdout for metrics
    parsed = parse_stdout_for_metrics(result["stdout"])

    # if training_acc was not found in checkpoint, use parsed model_accuracy from CLI output
    model_accuracy = training_acc if training_acc is not None else parsed.get("model_accuracy")

    # if infection percentage not found in stdout, fallback to 0.0
    infection_pct = parsed.get("infection_percentage")
    top1_conf = parsed.get("top1_confidence")

    # if predictions CSV exists, you can compute evaluation accuracy from it as fallback (optional)
    if model_accuracy is None and Path(PREDICTIONS_CSV).exists():
        try:
            import pandas as pd
            df = pd.read_csv(PREDICTIONS_CSV)
            # compute accuracy if columns exist
            if "true_label" in df.columns and "pred_label" in df.columns:
                model_accuracy = float((df["true_label"] == df["pred_label"]).mean() * 100.0)
        except Exception:
            pass

    recommended = recommend_action(infection_pct)

    return {
        # "model_accuracy": None if model_accuracy is None else round(float(model_accuracy), 2),
        # "top1_confidence": None if top1_conf is None else round(float(top1_conf), 2),
        # "infection_percentage": None if infection_pct is None else round(float(infection_pct), 2),
        # "recommended_action": recommended,
        "stdout": result["stdout"],   # raw text for debug
        "stderr": result["stderr"]    # raw text for debug
    }

# End of file

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Intelligent Sprinkler Backend...")
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
