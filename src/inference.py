import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.model import build_model

def load_classes(path):
    with open(path, "r") as f:
        return [c.strip() for c in f.readlines()]

def load_image(path, img_size):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)

def load_model(checkpoint, num_classes, device):
    model = build_model(num_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state['state_dict'])
    model.eval()
    model.to(device)
    return model

def compute_infection_percentage(pred_label: str):
    return 0.0 if "healthy" in pred_label.lower() else 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes_file", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cpu")

    classes = load_classes(args.classes_file)
    img = load_image(args.image_path, args.img_size)
    model = load_model(args.checkpoint, len(classes), device)

    with torch.no_grad():
        logits = model(img.to(device))
        probs = F.softmax(logits, dim=1)[0]

    top_prob, top_idx = torch.max(probs, dim=0)
    pred_label = classes[top_idx]

    infection_percentage = compute_infection_percentage(pred_label)

    print(f"\nModel accuracy from training: 99.54%")  # You told me this value
    print(f"Top-1 confidence: {top_prob.item()*100:.2f}%")
    print(f"Infection percentage: {infection_percentage:.2f}%\n")

if __name__ == "__main__":
    main()
