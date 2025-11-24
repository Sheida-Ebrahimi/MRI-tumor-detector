from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# Rebuilding model
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # single logit output

# Load state dict
state_dict = torch.load(
    "brain_tumor_resnet18.pt", 
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()

# Recreate model architecture
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Process uploaded image
def predict(img):
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)        # shape: [1, 1]
        prob = torch.sigmoid(logits)[0].item()  # convert to float

    return prob


# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file.stream)

        tumor_prob = predict(img)

        if tumor_prob >= 0.5:
            result = f"TUMOR DETECTED (confidence: {tumor_prob:.2f})"
        else:
            result = f"NO TUMOR (confidence: {(1 - tumor_prob):.2f})"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
