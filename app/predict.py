import json
import os
import re
import torch
import torch.nn as nn
from PIL import ImageOps
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = json.load(open(os.path.join(HERE, "classes.json")))
WEIGHTS = os.path.join(HERE, "resnet18_fruit17.pt")
CONFIDENCE_THRESHOLD = 0.60   # below this, the app says "not sure"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    return model.to(device).eval()


model = load_model()


def pretty(name):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name)      # "FreshApple" -> "Fresh Apple"


def predict(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu()

    scores = {pretty(CLASSES[i]): float(probs[i]) for i in range(len(CLASSES))}
    conf, idx = probs.max(0)
    conf, name = conf.item(), CLASSES[idx.item()]

    if name == "NotFruit":
        verdict = f"This doesn't look like a fruit ({conf:.0%})."
    elif conf < CONFIDENCE_THRESHOLD:
        verdict = f"Not sure ({conf:.0%} for {pretty(name)}). Try a closer photo of a single fruit."
    else:
        verdict = f"{pretty(name)} ({conf:.0%})"
    return verdict, scores