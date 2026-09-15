# src/predict_and_infect.py
"""
Clean version: NO model accuracy printed, NO extra logs.
Outputs ONLY:
{
  "top1_confidence": float,
  "infection_percentage": float,
  "recommended_action": str
}
"""

import argparse
import os
import json
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

# ----------------------
# Helpers
# ----------------------
def load_classes(classes_file: str) -> List[str]:
    with open(classes_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_model(num_classes: int):
    from torchvision import models
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path: str, num_classes: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = build_model(num_classes).to(device)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def transform_image(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])


def predict(model, image_path, transform, device, topk=3):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = probs.argsort()[::-1][:topk]
    return top_indices, probs


def infection_percentage(probs, classes):
    infected = 0.0
    for i, p in enumerate(probs):
        cname = classes[i].lower()
        if ("healthy" in cname) or ("background" in cname):
            continue
        infected += float(p)
    return infected * 100.0


def action_for(infect_pct: float) -> str:
    if infect_pct >= 70:
        return "High: Immediate treatment recommended (spray/isolate)."
    if infect_pct >= 30:
        return "Medium: Treat soon — targeted spraying suggested."
    if infect_pct >= 5:
        return "Low: Monitor or spot treatment."
    return "None: No treatment needed."


# ----------------------
# Main (only JSON output)
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes_file", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cpu")
    classes = load_classes(args.classes_file)
    model = load_checkpoint(args.checkpoint, len(classes), device)

    transform = transform_image(args.img_size)

    top_idx, probs = predict(model, args.image_path, transform, device, args.topk)

    # top-1 confidence
    top1_conf = float(probs[top_idx[0]] * 100.0)

    # infection percentage
    inf_pct = float(infection_percentage(probs, classes))

    # recommended action
    action = action_for(inf_pct)

    # RETURN ONLY CLEAN JSON
    result = {
        "top1_confidence": round(top1_conf, 2),
        "infection_percentage": round(inf_pct, 2),
        "recommended_action": action
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
