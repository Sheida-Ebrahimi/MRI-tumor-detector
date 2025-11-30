from flask import Flask, render_template, request
import os

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

import numpy as np

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)

# ----------------- Model setup -----------------

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # binary classifier: tumor vs no tumor

state_dict = torch.load("brain_tumor_resnet18.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(img_pil: Image.Image) -> float:
    """Return probability of 'tumor' (between 0 and 1)."""
    img_pil = img_pil.convert("RGB")
    x = transform(img_pil).unsqueeze(0)  # shape [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(x)              # shape [1, 1]
        prob = torch.sigmoid(logits)[0].item()

    return prob


def generate_gradcam(img_pil: Image.Image, save_path: str):
    """
    Generate a Grad-CAM heatmap for the given PIL image and save it to save_path.
    Highlights regions contributing to 'tumor' prediction.
    """
    img_pil = img_pil.convert("RGB")
    img_resized = img_pil.resize((224, 224))

    # RGB image as float32 in [0,1] for overlay
    rgb_np = np.array(img_resized).astype(np.float32) / 255.0

    # Tensor for the model
    input_tensor = transform(img_resized).unsqueeze(0)

    target_layer = model.layer4[-1]  # last conv block of ResNet18

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)

    # For binary output (single logit), default target is fine
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # H x W

    visualization = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    # Save the resulting heatmap image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vis_img = Image.fromarray(visualization)
    vis_img.save(save_path)


# ----------------- Flask route -----------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    heatmap_url = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            # Ensure uploads directory exists
            os.makedirs("static/uploads", exist_ok=True)
            os.makedirs("static/heatmaps", exist_ok=True)

            # Save original uploaded image
            filename = file.filename
            # (Optionally sanitize filename; for demo, we keep it simple)
            upload_path = os.path.join("static", "uploads", filename)
            file.save(upload_path)

            # Open image for prediction + Grad-CAM
            img = Image.open(upload_path)

            # Run prediction
            tumor_prob = predict(img)

            if tumor_prob >= 0.5:
                result = f"TUMOR DETECTED (confidence: {tumor_prob:.2f})"
            else:
                result = f"NO TUMOR (confidence: {(1 - tumor_prob):.2f})"

            # Build Grad-CAM heatmap and save
            heatmap_filename = os.path.splitext(filename)[0] + "_cam.png"
            heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
            generate_gradcam(img, heatmap_path)

            # URLs for template
            image_url = upload_path           # static/uploads/...
            heatmap_url = heatmap_path        # static/heatmaps/...

        else:
            result = "No file uploaded. Please choose an image."

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        heatmap_url=heatmap_url
    )


if __name__ == "__main__":
    app.run(debug=True)
