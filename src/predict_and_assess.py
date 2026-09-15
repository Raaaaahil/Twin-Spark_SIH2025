# src/predict_and_assess.py
# Full script — copy-paste and save.

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import importlib

def load_classes(classes_file):
    with open(classes_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

def build_model_from_src(num_classes, device):
    """
    Try to load model factory from src.model if available, otherwise build resnet18.
    """
    try:
        mod = importlib.import_module("src.model")
        # try several function/class names
        if hasattr(mod, "build_model"):
            model = mod.build_model(num_classes)
        elif hasattr(mod, "get_model"):
            model = mod.get_model(num_classes)
        elif hasattr(mod, "PlantModel"):
            model = mod.PlantModel(num_classes)
        elif hasattr(mod, "Model"):
            model = mod.Model(num_classes)
        else:
            raise Exception("src.model exists but no recognized build function/class found.")
        return model.to(device)
    except Exception:
        # fallback to torchvision resnet18
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)

def load_checkpoint(checkpoint_path, num_classes, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    # If saved full nn.Module object:
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model, {}
    # If saved as dict (common)
    if isinstance(ckpt, dict):
        # find nested state dict
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # if state is the dict with keys like 'epoch','optimizer_state_dict'
        if not isinstance(state, dict):
            # fallback: treat ckpt as state dict
            state = ckpt
        # remove possible 'module.' prefix
        new_state = {}
        for k, v in state.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        model = build_model_from_src(num_classes, device)
        # try loading state dict
        try:
            model.load_state_dict(new_state)
        except RuntimeError as e:
            # try if the checkpoint actually contains a 'model_state_dict' but different naming:
            # attempt to extract 'model_state_dict' again
            raise RuntimeError(f"Failed loading state_dict: {e}")
        model.to(device)
        model.eval()
        meta = {k:v for k,v in ckpt.items() if k not in ("model_state_dict","state_dict")}
        return model, meta
    # unknown format
    raise RuntimeError("Unsupported checkpoint format")

def preprocess_image(img_path, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)  # batch dim

def compute_infection_and_topk(probs, classes, topk=3):
    # probs: 1D numpy or tensor of probabilities summing to 1
    probs = probs.detach().cpu().numpy().squeeze()
    order = probs.argsort()[::-1]
    topk_idxs = order[:topk]
    topk_items = [(classes[i], float(probs[i])) for i in topk_idxs]
    # define healthy criterion: label contains 'healthy' (case-insensitive)
    healthy_mask = [("healthy" in c.lower() or "background" in c.lower()) for c in classes]
    # infection percentage: sum of probabilities of classes that are NOT healthy
    infection = float(probs[~(pd.Series(healthy_mask).values)].sum())
    # top1_conf:
    top1_conf = float(probs[topk_idxs[0]])
    return infection * 100.0, top1_conf * 100.0, topk_items

def dataset_accuracy_from_csv(predictions_csv, labels_csv=None):
    # predictions_csv expected to have columns: image,true_label,pred_label,...
    if not os.path.exists(predictions_csv):
        return None
    df = pd.read_csv(predictions_csv)
    # try finding columns:
    if 'true_label' in df.columns and 'pred_label' in df.columns:
        correct = (df['true_label'] == df['pred_label']).sum()
        total = len(df)
        return (correct / total) * 100.0
    # if only image and top1 present, try other heuristics
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes_file", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--predictions_csv", default="results/predictions.csv")
    parser.add_argument("--labels_csv", default="data/labels.csv", help="optional: used for dataset accuracy if present")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.classes_file)
    num_classes = len(classes)
    print(f"Loaded {num_classes} classes")

    print(f"Loading model from: {args.checkpoint}")
    model, meta = load_checkpoint(args.checkpoint, num_classes, device)
    print("Model loaded and in eval() mode\n")

    # make prediction
    x = preprocess_image(args.image_path, args.img_size).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    infection_pct, top1_conf, topk_items = compute_infection_and_topk(probs, classes, topk=args.topk)

    # attempt to get model accuracy from checkpoint metadata or from predictions CSV
    model_accuracy = None
    # first try metadata from checkpoint (common key names)
    for k in ("best_acc", "accuracy", "acc", "train_acc", "val_acc"):
        if isinstance(meta.get(k), (int, float)):
            model_accuracy = float(meta[k])
            break
    # then try predictions CSV
    if model_accuracy is None:
        model_accuracy = dataset_accuracy_from_csv(args.predictions_csv, args.labels_csv)

    # print exactly requested outputs (clean)
    if model_accuracy is None:
        print("Model accuracy from checkpoint/CSV: Not available")
    # else:
    #     print(f"Model accuracy from checkpoint/CSV: {model_accuracy:.2f}%")
    print(f"Top-1 confidence: {top1_conf:.2f}%")
    print(f"Infection percentage: {infection_pct:.2f}%\n")

    # recommended action by infection level
    if infection_pct >= 70:
        action = "High: Treat immediately with full-strength pesticide/spraying."
    elif infection_pct >= 30:
        action = "Medium: Apply targeted treatment and monitor closely."
    elif infection_pct > 0:
        action = "Low: Monitor and consider treatment only if spreads."
    else:
        action = "No treatment needed. Monitor the plant."
    print("Recommended action:", action)

    # optional: print topk
    print("\nTop predictions:")
    for i,(label,prob) in enumerate(topk_items, start=1):
        print(f"Top {i}: {label} ({prob*100.0 if prob<=1.01 else prob:.2f}%)")

if __name__ == "__main__":
    main()
