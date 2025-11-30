from flask import Flask, render_template, request
import os

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

state_dict = torch.load("brain_tumor_resnet18.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(img_pil):
    img_pil = img_pil.convert("RGB")
    x = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0].item()

    return prob


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            filename = file.filename

            upload_path = os.path.join("static", "uploads", filename)
            os.makedirs("static/uploads", exist_ok=True)
            file.save(upload_path)

            img = Image.open(upload_path)
            tumor_prob = predict(img)

            if tumor_prob >= 0.5:
                result = f"TUMOR DETECTED (confidence: {tumor_prob:.2f})"
            else:
                result = f"NO TUMOR (confidence: {(1 - tumor_prob):.2f})"

            image_url = upload_path

    return render_template(
        "index.html",
        result=result,
        image_url=image_url
    )


if __name__ == "__main__":
    app.run(debug=True)
