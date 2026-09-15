# src/predict_and_analyze.py
"""
Usage (example):
python -m src.predict_and_analyze \
  --image_path test_images/sample.jpg \
  --checkpoint checkpoints/epoch_30.pth \
  --classes_file data/classes.txt \
  --img_size 224 \
  --predictions_csv results/predictions.csv \
  --labels_csv data/labels.csv

This script:
- Loads checkpoint robustly (state_dict or full model).
- Runs inference on a single image.
- Prints:
   1) Model accuracy (from checkpoint metadata if available; else from predictions CSV if present and matches labels).
   2) Top-1 confidence.
   3) Infection percentage (sum of probs for non-healthy/non-background classes).
- Prints recommended action (Low/Medium/High).
"""
import argparse
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd

# --- robust model loader (state_dict or full model) ---
import importlib

def _build_model_from_src(num_classes, device):
    # Try to import src.model and create model automatically from common names
    mod = importlib.import_module("src.model")
    if hasattr(mod, "build_model"):
        model = mod.build_model(num_classes)
    elif hasattr(mod, "get_model"):
        model = mod.get_model(num_classes)
    elif hasattr(mod, "PlantModel"):
        model = mod.PlantModel(num_classes)
    elif hasattr(mod, "Model"):
        model = mod.Model(num_classes)
    else:
        # fallback: try common torchvision resnet18 if src.model absent
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        except Exception as e:
            raise ImportError("Cannot build model automatically. Edit _build_model_from_src() to match your src.model file.") from e
    return model.to(device)

def load_model(checkpoint_path: str, num_classes: int, device=torch.device("cpu")):
    ckpt = torch.load(checkpoint_path, map_location=device)

    # If checkpoint is a full model object
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        model.to(device)
        model.eval()
        return model, {}

    # If checkpoint is a dict (state_dict or dict with 'model_state_dict' etc.)
    if isinstance(ckpt, dict):
        # keep metadata if present
        meta = {}
        meta['raw_keys'] = list(ckpt.keys())
        # common conventions:
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # if checkpoint contains training metadata (epoch, best_acc, class_to_idx)
        for k in ('epoch','best_acc','accuracy','model_state_dict','state_dict','class_to_idx'):
            if k in ckpt:
                meta[k] = ckpt[k]

        # If state_dict is a nested dict mapping strings->tensors
        if isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
            model = _build_model_from_src(num_classes, device)
            # strip "module." prefixes if present
            new_state = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_key] = v
            model.load_state_dict(new_state, strict=False)
            model.to(device)
            model.eval()
            return model, meta

        # fallback: unknown dict (maybe entire model saved as dict) - try to load keys
        try:
            ckpt.eval()
            return ckpt, meta
        except Exception:
            raise RuntimeError("Unsupported checkpoint format.")
    # unknown type
    raise RuntimeError("Unsupported checkpoint type.")

# --- transforms (match your training: resize, center crop, normalize) ---
def build_transform(img_size=224):
    # Typical ImageNet-like normalization — if training used different, change here
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform

def infer_image(model, device, classes, image_path, img_size=224, topk=3):
    img = Image.open(image_path).convert("RGB")
    transform = build_transform(img_size)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    # sort
    idxs = probs.argsort()[::-1][:topk]
    preds = [(classes[i], float(probs[i])) for i in idxs]
    return preds, probs

def determine_infection(probs, classes):
    # Define healthy/background classes heuristically:
    # Treat classes that end with 'healthy' (case-insensitive) or contain 'Background' as non-infected.
    non_infected_idx = []
    for i, c in enumerate(classes):
        lc = c.lower()
        if 'healthy' in lc or 'background' in lc:
            non_infected_idx.append(i)
    # infection percentage = sum of probs of all classes NOT in non_infected_idx
    infected_prob = float(probs.sum() - probs[non_infected_idx].sum()) if len(non_infected_idx) > 0 else float(probs.sum())
    # safety clamp:
    infected_prob = max(0.0, min(1.0, infected_prob))
    return infected_prob

def compute_accuracy_from_csv(predictions_csv, labels_csv):
    # Both CSVs should use columns 'image' or 'image_path' (predictions: image path + pred label),
    # labels_csv should have 'image_path' relative to data root and 'label' columns as your labels.csv format.
    try:
        df_pred = pd.read_csv(predictions_csv)
        df_lab = pd.read_csv(labels_csv)
    except Exception as e:
        return None

    # normalize column names
    pred_cols = df_pred.columns.str.lower()
    lab_cols = df_lab.columns.str.lower()
    # find image column
    img_col_pred = None
    for c in df_pred.columns:
        if c.lower().startswith('image') or c.lower().startswith('image_path') or c.lower().startswith('image,'):
            img_col_pred = c
            break
    if img_col_pred is None:
        # try first column
        img_col_pred = df_pred.columns[0]
    # find predicted label column
    pred_label_col = None
    for c in df_pred.columns:
        if 'pred' in c.lower() and 'label' in c.lower():
            pred_label_col = c
            break
    if pred_label_col is None:
        # fallback to second column
        if len(df_pred.columns) >= 2:
            pred_label_col = df_pred.columns[1]
        else:
            return None

    # labels file columns
    label_img_col = None
    label_label_col = None
    for c in df_lab.columns:
        if 'image' in c.lower():
            label_img_col = c
        if 'label' in c.lower():
            label_label_col = c
    if label_img_col is None or label_label_col is None:
        return None

    # Merge by file name base (basename) to be more flexible
    df_pred['fname'] = df_pred[img_col_pred].apply(lambda p: os.path.basename(str(p)).lower())
    df_lab['fname'] = df_lab[label_img_col].apply(lambda p: os.path.basename(str(p)).lower())

    merged = pd.merge(df_pred, df_lab, on='fname', how='inner', suffixes=('_pred','_lab'))
    if merged.shape[0] == 0:
        return None
    # compare predicted label and true label (strings)
    # find pred label column and true label column names in merged:
    pred_col = pred_label_col
    true_col = label_label_col
    # sometimes predictions have format "image,true_label,pred_label,..." - try to find 'pred_label' column explicitly:
    for c in merged.columns:
        if 'pred_label' in c.lower():
            pred_col = c
        if c.lower().endswith('_label') and c != true_col:
            # skip confusion; keep original
            pass
    # compute accuracy
    # normalize strings
    merged['pred_norm'] = merged[pred_col].astype(str).str.lower().str.strip()
    merged['true_norm'] = merged[true_col].astype(str).str.lower().str.strip()
    correct = (merged['pred_norm'] == merged['true_norm']).sum()
    acc = float(correct) / merged.shape[0]
    return acc * 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes_file", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--predictions_csv", default="results/predictions.csv")
    parser.add_argument("--labels_csv", default="data/labels.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load classes
    with open(args.classes_file, "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f.readlines() if l.strip()]
    print(f"Loaded {len(classes)} classes")

    # load model
    print(f"Loading model from: {args.checkpoint}")
    model, meta = load_model(args.checkpoint, len(classes), device)
    print("Model loaded and in eval() mode\n")

    # attempt to get model accuracy from checkpoint metadata
    model_accuracy = None
    if 'best_acc' in meta:
        model_accuracy = float(meta.get('best_acc')) if meta.get('best_acc') is not None else None
    elif 'accuracy' in meta:
        model_accuracy = float(meta.get('accuracy'))
    elif 'epoch' in meta and 'model_state_dict' in meta.get('raw_keys', []):
        # no clear accuracy stored
        model_accuracy = None

    # if no metadata accuracy, try to compute from predictions CSV & labels CSV
    if model_accuracy is None:
        acc_from_csv = compute_accuracy_from_csv(args.predictions_csv, args.labels_csv)
        if acc_from_csv is not None:
            model_accuracy = acc_from_csv
    # final fallback: unknown
    if model_accuracy is None:
        model_accuracy_text = "Unknown (no metadata or matching predictions/labels)"
    else:
        model_accuracy_text = f"{model_accuracy:.2f}%"

    # infer single image
    preds, probs = infer_image(model, device, classes, args.image_path, img_size=args.img_size, topk=args.topk)
    top1_label, top1_prob = preds[0]
    top1_conf = top1_prob * 100.0

    infected_prob = determine_infection(probs, classes) * 100.0

    # Print only what you asked for
    print(f"Model accuracy (from checkpoint/CSV): {model_accuracy_text}")
    print(f"Top-1 confidence: {top1_conf:.2f}%")
    print(f"Infection percentage: {infected_prob:.2f}%\n")

    # Simple threshold-based recommended action:
    if infected_prob >= 60:
        action = "HIGH: Spray immediately with full dose."
    elif infected_prob >= 25:
        action = "MEDIUM: Treat soon; consider targeted spray."
    else:
        action = "LOW: No treatment needed; monitor plant."
    print("Recommended action:", action)

    # Also print top-k for debugging
    print("\nTop predictions:")
    for i, (lbl, p) in enumerate(preds, 1):
        print(f"Top {i}: {lbl} ({p*100:.2f}%)")

if __name__ == "__main__":
    main()
